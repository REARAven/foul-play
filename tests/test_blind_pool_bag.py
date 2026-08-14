from __future__ import annotations

import copy
import errno
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import traceback
from unittest import mock
import unittest

from fp.data.blind_pool import (
    PRIVATE_ROOT_ENV,
    REGISTRY_PATH_ENV,
    STATE_PATH_ENV,
    BlindPoolBagStore,
    BlindPoolStateConfig,
    BlindPoolValidationError,
    compute_registry_fingerprint,
    load_blind_pool_config,
    load_blind_pool_registry,
    load_blind_pool_state_config,
)
from fp.data.blind_pool.locking import BlindPoolStateLock
from fp.data.blind_pool.state import _fsync_directory


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SENTINEL = "PHASE3-SYNTHETIC-PRIVATE-SENTINEL"


class IdentityRandom:
    def shuffle(self, values: list[str]) -> None:
        return None

    def randrange(self, start: int, stop: int | None = None) -> int:
        return start


class BoundaryCollisionRandom:
    def __init__(self) -> None:
        self.calls = 0

    def shuffle(self, values: list[str]) -> None:
        self.calls += 1
        if self.calls > 1:
            values.reverse()

    def randrange(self, start: int, stop: int | None = None) -> int:
        return start


class FailingTextStream:
    def __init__(self, stream, failure_point: str) -> None:
        self._stream = stream
        self._failure_point = failure_point

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        return self._stream.__exit__(exc_type, exc, traceback_value)

    def write(self, value: str):
        if self._failure_point == "write":
            raise OSError("synthetic write failure")
        return self._stream.write(value)

    def flush(self) -> None:
        if self._failure_point == "flush":
            raise OSError("synthetic flush failure")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.kill()
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.wait(timeout=2)


class BlindPoolBagFixture(unittest.TestCase):
    active_ids = ("BL-001-v1", "BL-002-v1", "BL-003-v1")

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.private_root = self.base / ("private-" + PRIVATE_SENTINEL)
        self.team_directory = self.private_root / "teams"
        self.state_directory = self.private_root / "state"
        self.team_directory.mkdir(parents=True)
        self.state_directory.mkdir()
        self.registry_path = self.private_root / "registry.json"
        self.state_path = self.state_directory / "bag-state.json"
        self._write_registry()
        self.environment = {
            PRIVATE_ROOT_ENV: str(self.private_root),
            REGISTRY_PATH_ENV: str(self.registry_path),
            STATE_PATH_ENV: str(self.state_path),
        }
        self.pool_config = load_blind_pool_config(
            self.environment,
            repository_root=ROOT,
        )
        self.state_config = load_blind_pool_state_config(
            self.environment,
            repository_root=ROOT,
        )
        self.registry = load_blind_pool_registry(self.pool_config)

    def _entry(self, team_id: str, active: bool = True) -> dict[str, object]:
        payload = (PRIVATE_SENTINEL + "-" + team_id + "\n").encode()
        team_path = self.team_directory / (team_id + ".team")
        team_path.write_bytes(payload)
        return {
            "team_id": team_id,
            "active": active,
            "team_file": "teams/" + team_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _registry_document(
        self,
        *,
        active_ids: tuple[str, ...] | None = None,
        registry_version: str = "3.0",
        include_inactive: bool = True,
    ) -> dict[str, object]:
        ids = self.active_ids if active_ids is None else active_ids
        entries = [self._entry(team_id) for team_id in ids]
        if include_inactive:
            entries.append(self._entry("BL-999-v1", active=False))
        return {
            "schema_version": 1,
            "registry_version": registry_version,
            "format_id": "gen9tugs",
            "entries": entries,
        }

    def _write_registry(self, **kwargs) -> dict[str, object]:
        document = self._registry_document(**kwargs)
        self.registry_path.write_text(
            json.dumps(document, separators=(",", ":")),
            encoding="utf-8",
        )
        return document

    def reload_registry(self):
        self.registry = load_blind_pool_registry(self.pool_config)
        return self.registry

    def store(
        self,
        *,
        random_source=None,
        reservation_ids: list[str] | None = None,
        lock_timeout_seconds: float = 1.0,
        registry=None,
    ) -> BlindPoolBagStore:
        identifiers = iter(
            reservation_ids
            or [format(index, "032x") for index in range(1, 100)]
        )
        return BlindPoolBagStore(
            self.state_config,
            self.registry if registry is None else registry,
            random_source=random_source or IdentityRandom(),
            reservation_id_factory=lambda: next(identifiers),
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def state_document(self) -> dict[str, object]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state_document(self, document: dict[str, object]) -> None:
        self.state_path.write_text(
            json.dumps(document, separators=(",", ":")),
            encoding="utf-8",
        )

    def assert_error(self, code: str, operation) -> BlindPoolValidationError:
        with self.assertRaises(BlindPoolValidationError) as captured:
            operation()
        self.assertEqual(code, captured.exception.code)
        return captured.exception

    def consume(self, store: BlindPoolBagStore):
        reservation = store.reserve_next()
        store.commit_reservation(reservation.reservation_id)
        return reservation


class TestBlindPoolBagConfiguration(BlindPoolBagFixture):
    def test_package_import_has_no_config_or_filesystem_side_effects(self):
        before = {
            "registry": self.registry_path.read_bytes(),
            "teams": tuple(
                sorted(
                    (path.name, path.read_bytes())
                    for path in self.team_directory.iterdir()
                )
            ),
        }
        completed = subprocess.run(
            [sys.executable, "-B", "-c", "import fp.data.blind_pool"],
            cwd=ROOT,
            env=dict(os.environ, **self.environment),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertFalse(self.state_path.exists())
        self.assertFalse(self.state_config.lock_path.exists())
        self.assertEqual(before["registry"], self.registry_path.read_bytes())
        self.assertEqual(
            before["teams"],
            tuple(
                sorted(
                    (path.name, path.read_bytes())
                    for path in self.team_directory.iterdir()
                )
            ),
        )

    def test_valid_external_state_path_and_absent_file_are_accepted(self):
        self.assertIsNotNone(self.state_config)
        self.assertEqual(self.state_path.resolve(), self.state_config.state_path)
        self.assertFalse(self.state_path.exists())
        self.assertTrue(self.state_config.lock_path.parent.is_dir())

    def test_relative_state_path_is_rejected(self):
        self.assert_error(
            "state_config_path_invalid",
            lambda: BlindPoolStateConfig(self.pool_config, Path("relative.json")),
        )

    def test_non_path_state_value_is_rejected_safely(self):
        error = self.assert_error(
            "state_config_path_invalid",
            lambda: BlindPoolStateConfig(self.pool_config, object()),
        )
        self.assertNotIn(PRIVATE_SENTINEL, str(error))

    def test_in_repository_state_path_is_rejected(self):
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(ROOT / "state.json")},
        )
        self.assert_error(
            "state_path_escape",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_state_outside_private_root_is_rejected(self):
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(self.base / "outside.json")},
        )
        self.assert_error(
            "state_path_escape",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_state_equal_to_registry_is_rejected(self):
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(self.registry_path)},
        )
        self.assert_error(
            "state_registry_collision",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_lock_equal_to_registry_is_rejected_without_path_disclosure(self):
        collision_registry = self.state_config.lock_path
        collision_registry.write_bytes(self.registry_path.read_bytes())
        environment = dict(
            self.environment,
            **{REGISTRY_PATH_ENV: str(collision_registry)},
        )
        error = self.assert_error(
            "state_lock_registry_collision",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )
        self.assertNotIn(str(collision_registry), str(error))

    def test_state_equal_to_registered_team_file_is_rejected_safely(self):
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(self.registry.entries[0].resolved_team_path)},
        )
        state_config = load_blind_pool_state_config(
            environment,
            repository_root=ROOT,
        )
        error = self.assert_error(
            "state_team_file_collision",
            lambda: BlindPoolBagStore(state_config, self.registry),
        )
        self.assertNotIn(str(state_config.state_path), str(error))

    def test_registry_equal_to_registered_team_file_is_rejected_safely(self):
        first = self.registry.entries[0]
        registry = replace(
            self.registry,
            entries=(
                replace(first, resolved_team_path=self.registry_path.resolve()),
                *self.registry.entries[1:],
            ),
        )
        error = self.assert_error(
            "registry_team_file_collision",
            lambda: BlindPoolBagStore(self.state_config, registry),
        )
        self.assertNotIn(str(self.registry_path), str(error))

    def test_lock_equal_to_registered_team_file_is_rejected_safely(self):
        collision_state = self.state_directory / "collision-state.json"
        collision_lock = collision_state.with_name(collision_state.name + ".lock")
        payload = (PRIVATE_SENTINEL + "-collision\n").encode()
        collision_lock.write_bytes(payload)
        document = self._registry_document()
        document["entries"][0]["team_file"] = "state/" + collision_lock.name
        document["entries"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
        self.registry_path.write_text(json.dumps(document), encoding="utf-8")
        registry = self.reload_registry()
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(collision_state)},
        )
        state_config = load_blind_pool_state_config(
            environment,
            repository_root=ROOT,
        )
        error = self.assert_error(
            "state_lock_team_file_collision",
            lambda: BlindPoolBagStore(state_config, registry),
        )
        self.assertNotIn(str(collision_lock), str(error))

    def test_transaction_rejects_state_path_identity_change(self):
        store = self.store()
        changed = BlindPoolStateConfig(
            self.state_config.pool_config,
            self.state_directory / "changed-state.json",
        )
        with mock.patch(
            "fp.data.blind_pool.bag.validate_blind_pool_state_config",
            return_value=changed,
        ):
            self.assert_error("state_path_changed", store.initialize_or_load)
        self.assertFalse(self.state_path.exists())

    def test_transaction_rejects_lock_path_identity_change(self):
        store = self.store()
        changed_lock = self.state_directory / "changed-state.lock"
        with mock.patch.object(
            store,
            "_resolve_lock_path",
            return_value=changed_lock,
        ):
            self.assert_error("state_lock_path_changed", store.initialize_or_load)
        self.assertFalse(self.state_path.exists())

    def test_missing_parent_is_rejected(self):
        missing = self.private_root / "missing" / "state.json"
        environment = dict(self.environment, **{STATE_PATH_ENV: str(missing)})
        self.assert_error(
            "state_parent_unavailable",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_parent_that_is_not_a_directory_is_rejected(self):
        parent = self.private_root / "not-a-directory"
        parent.write_text("synthetic", encoding="utf-8")
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(parent / "state.json")},
        )
        self.assert_error(
            "state_parent_invalid",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_existing_non_file_state_path_is_rejected(self):
        self.state_path.mkdir()
        self.assert_error(
            "state_file_invalid",
            lambda: load_blind_pool_state_config(
                self.environment,
                repository_root=ROOT,
            ),
        )

    def test_existing_regular_state_and_empty_lock_file_are_accepted(self):
        self.state_path.write_text("{}", encoding="utf-8")
        self.state_config.lock_path.touch()
        configured = load_blind_pool_state_config(
            self.environment,
            repository_root=ROOT,
        )
        self.assertEqual(self.state_path.resolve(), configured.state_path)
        self.assertTrue(configured.lock_path.is_file())

    def test_symlink_escape_is_rejected_where_supported(self):
        outside = self.base / "external-state"
        outside.mkdir()
        link = self.private_root / "linked-state"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks are unavailable")
        environment = dict(
            self.environment,
            **{STATE_PATH_ENV: str(link / "bag.json")},
        )
        self.assert_error(
            "state_path_escape",
            lambda: load_blind_pool_state_config(environment, repository_root=ROOT),
        )

    def test_state_configuration_is_dormant_when_absent(self):
        phase_two_environment = {
            PRIVATE_ROOT_ENV: str(self.private_root),
            REGISTRY_PATH_ENV: str(self.registry_path),
        }
        with mock.patch(
            "fp.data.blind_pool.config._resolve_state_path",
            side_effect=AssertionError("state path must remain dormant"),
        ):
            self.assertIsNone(
                load_blind_pool_state_config(
                    phase_two_environment,
                    repository_root=ROOT,
                )
            )
        self.assertIsNotNone(
            load_blind_pool_config(
                phase_two_environment,
                repository_root=ROOT,
            )
        )

    def test_state_without_registry_configuration_is_rejected_safely(self):
        self.assert_error(
            "state_config_incomplete",
            lambda: load_blind_pool_state_config(
                {STATE_PATH_ENV: str(self.state_path)}
            ),
        )

    def test_state_config_repr_hides_all_paths(self):
        rendered = repr(self.state_config)
        self.assertEqual("BlindPoolStateConfig(configured=True)", rendered)
        self.assertNotIn(str(self.private_root), rendered)
        self.assertNotIn(str(self.state_path), rendered)


class TestBlindPoolBagInitialization(BlindPoolBagFixture):
    def test_first_initialization_persists_every_active_id_once(self):
        state = self.store().initialize_or_load()
        self.assertEqual(1, state.schema_version)
        self.assertEqual(1, state.cycle_number)
        self.assertEqual(0, state.next_index)
        self.assertIsNone(state.last_consumed_id)
        self.assertIsNone(state.reservation)
        self.assertCountEqual(self.active_ids, state.cycle_order)
        self.assertEqual(len(self.active_ids), len(set(state.cycle_order)))
        self.assertNotIn("BL-999-v1", state.cycle_order)
        self.assertTrue(self.state_path.is_file())

    def test_initial_state_reloads_identically(self):
        initial = self.store().initialize_or_load()
        restarted = self.store(
            random_source=BoundaryCollisionRandom()
        ).initialize_or_load()
        self.assertEqual(initial, restarted)

    def test_two_active_entries_are_accepted(self):
        self._write_registry(active_ids=self.active_ids[:2], include_inactive=False)
        registry = self.reload_registry()
        state = self.store(registry=registry).initialize_or_load()
        self.assertCountEqual(self.active_ids[:2], state.cycle_order)

    def test_one_active_entry_is_rejected_only_by_bag_layer(self):
        self._write_registry(active_ids=self.active_ids[:1], include_inactive=False)
        registry = self.reload_registry()
        self.assertEqual(1, len(registry.active_entries))
        self.assert_error(
            "insufficient_active_entries",
            lambda: self.store(registry=registry),
        )

    def test_default_constructs_system_random(self):
        with mock.patch(
            "fp.data.blind_pool.bag.random.SystemRandom",
            wraps=__import__("random").SystemRandom,
        ) as system_random:
            BlindPoolBagStore(self.state_config, self.registry)
        system_random.assert_called_once_with()

    def test_state_and_reservation_models_are_immutable_and_safe(self):
        store = self.store()
        state = store.initialize_or_load()
        reservation = store.reserve_next()
        with self.assertRaises(FrozenInstanceError):
            state.next_index = 7
        rendered = repr(store) + repr(state) + repr(reservation)
        self.assertNotIn(str(self.private_root), rendered)
        self.assertNotIn(state.registry_fingerprint, rendered)
        self.assertNotIn(reservation.reservation_id, rendered)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)


class TestBlindPoolBagCycles(BlindPoolBagFixture):
    def test_no_repeat_and_every_id_once_within_cycle(self):
        store = self.store()
        state = store.initialize_or_load()
        consumed = [self.consume(store).team_id for _ in self.active_ids]
        self.assertEqual(list(state.cycle_order), consumed)
        self.assertEqual(len(consumed), len(set(consumed)))
        self.assertEqual(len(self.active_ids), store.snapshot().next_index)

    def test_new_cycle_only_after_prior_cycle_is_committed(self):
        store = self.store()
        store.initialize_or_load()
        first = store.reserve_next()
        self.assertEqual(1, first.cycle_number)
        store.release_reservation(first.reservation_id)
        self.assertEqual(1, store.snapshot().cycle_number)
        for _ in self.active_ids:
            self.consume(store)
        self.assertEqual(1, store.snapshot().cycle_number)
        next_cycle = store.reserve_next()
        self.assertEqual(2, next_cycle.cycle_number)

    def test_cycle_boundary_repeat_is_corrected_by_bounded_swap(self):
        source = BoundaryCollisionRandom()
        store = self.store(random_source=source)
        first_cycle = store.initialize_or_load().cycle_order
        for _ in first_cycle:
            self.consume(store)
        previous_last = store.snapshot().last_consumed_id
        next_reservation = store.reserve_next()
        self.assertNotEqual(previous_last, next_reservation.team_id)
        self.assertEqual(2, source.calls)

    def test_restart_mid_cycle_preserves_remaining_order(self):
        store = self.store()
        initial = store.initialize_or_load()
        self.consume(store)
        restarted = self.store(random_source=BoundaryCollisionRandom())
        snapshot = restarted.snapshot()
        self.assertEqual(initial.cycle_order, snapshot.cycle_order)
        self.assertEqual(1, snapshot.next_index)
        self.assertEqual(initial.cycle_order[1], restarted.reserve_next().team_id)

    def test_deterministic_injection_allows_exact_order_assertion(self):
        state = self.store(random_source=IdentityRandom()).initialize_or_load()
        self.assertEqual(self.active_ids, state.cycle_order)


class TestBlindPoolBagReservations(BlindPoolBagFixture):
    def test_reserve_persists_without_advancing(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        snapshot = store.snapshot()
        self.assertEqual(0, snapshot.next_index)
        self.assertEqual(reservation, snapshot.reservation)
        self.assertEqual("reserved", reservation.phase)

    def test_second_reserve_with_unresolved_reservation_fails(self):
        store = self.store()
        store.initialize_or_load()
        store.reserve_next()
        before = self.state_path.read_bytes()
        self.assert_error("unresolved_reservation_exists", store.reserve_next)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_restart_preserves_unresolved_reservation(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        restarted = self.store()
        self.assertEqual(reservation, restarted.snapshot().reservation)
        self.assert_error("unresolved_reservation_exists", restarted.reserve_next)

    def test_restart_can_explicitly_commit_unresolved_reservation(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        restarted = self.store()
        committed = restarted.commit_reservation(reservation.reservation_id)
        self.assertEqual(1, committed.next_index)
        self.assertEqual(reservation.team_id, committed.last_consumed_id)

    def test_restart_can_explicitly_release_unresolved_reservation(self):
        store = self.store(reservation_ids=["1" * 32])
        store.initialize_or_load()
        reservation = store.reserve_next()
        restarted = self.store(reservation_ids=["2" * 32])
        restarted.release_reservation(reservation.reservation_id)
        next_reservation = restarted.reserve_next()
        self.assertEqual(reservation.team_id, next_reservation.team_id)
        self.assertEqual(reservation.position, next_reservation.position)
        self.assertNotEqual(reservation.reservation_id, next_reservation.reservation_id)

    def test_commit_advances_exactly_once(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        committed = store.commit_reservation(reservation.reservation_id)
        self.assertEqual(1, committed.next_index)
        self.assertEqual(reservation.team_id, committed.last_consumed_id)
        self.assertIsNone(committed.reservation)
        before = self.state_path.read_bytes()
        self.assert_error(
            "reservation_not_found",
            lambda: store.commit_reservation(reservation.reservation_id),
        )
        self.assertEqual(before, self.state_path.read_bytes())

    def test_wrong_reservation_id_does_not_change_state(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        before = self.state_path.read_bytes()
        self.assert_error(
            "reservation_identity_mismatch",
            lambda: store.commit_reservation("f" * 32),
        )
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual(reservation, store.snapshot().reservation)

    def test_malformed_reservation_id_is_rejected_without_echo(self):
        store = self.store()
        store.initialize_or_load()
        store.reserve_next()
        error = self.assert_error(
            "reservation_id_invalid",
            lambda: store.commit_reservation(PRIVATE_SENTINEL),
        )
        self.assertNotIn(PRIVATE_SENTINEL, str(error))

    def test_reservation_id_factory_failure_is_sanitized(self):
        def fail():
            raise RuntimeError(PRIVATE_SENTINEL)

        store = BlindPoolBagStore(
            self.state_config,
            self.registry,
            random_source=IdentityRandom(),
            reservation_id_factory=fail,
        )
        store.initialize_or_load()
        error = self.assert_error(
            "reservation_id_generation_failed",
            store.reserve_next,
        )
        rendered = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        )
        self.assertNotIn(PRIVATE_SENTINEL, rendered)

    def test_release_does_not_advance_and_same_team_is_reserved_again(self):
        store = self.store(reservation_ids=["1" * 32, "2" * 32])
        state = store.initialize_or_load()
        first = store.reserve_next()
        released = store.release_reservation(first.reservation_id)
        self.assertEqual(0, released.next_index)
        self.assertEqual(state.cycle_order, released.cycle_order)
        self.assertIsNone(released.last_consumed_id)
        second = store.reserve_next()
        self.assertEqual(first.team_id, second.team_id)
        self.assertNotEqual(first.reservation_id, second.reservation_id)

    def test_successive_reservations_use_unique_ids(self):
        store = self.store(reservation_ids=["1" * 32, "2" * 32])
        store.initialize_or_load()
        first = store.reserve_next()
        store.commit_reservation(first.reservation_id)
        second = store.reserve_next()
        self.assertNotEqual(first.reservation_id, second.reservation_id)

    def test_wrong_release_identity_does_not_change_state(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        before = self.state_path.read_bytes()
        self.assert_error(
            "reservation_identity_mismatch",
            lambda: store.release_reservation("f" * 32),
        )
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual(reservation, store.snapshot().reservation)

    def test_duplicate_release_fails_without_change(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        store.release_reservation(reservation.reservation_id)
        before = self.state_path.read_bytes()
        self.assert_error(
            "reservation_not_found",
            lambda: store.release_reservation(reservation.reservation_id),
        )
        self.assertEqual(before, self.state_path.read_bytes())


class TestBlindPoolBagStateValidation(BlindPoolBagFixture):
    def initialized_document(self) -> dict[str, object]:
        self.store().initialize_or_load()
        return self.state_document()

    def assert_document_error(self, code: str, document: dict[str, object]) -> None:
        self.write_state_document(document)
        self.assert_error(code, self.store().snapshot)

    def test_unknown_and_missing_fields_are_rejected(self):
        valid = self.initialized_document()
        document = copy.deepcopy(valid)
        document["unknown"] = PRIVATE_SENTINEL
        self.assert_document_error("state_unexpected_field", document)
        document = copy.deepcopy(valid)
        del document["next_index"]
        self.assert_document_error("state_missing_required_field", document)

    def test_duplicate_json_fields_are_rejected(self):
        document = self.initialized_document()
        raw = json.dumps(document)
        self.state_path.write_text(raw[:-1] + ',"next_index":0}', encoding="utf-8")
        self.assert_error("state_duplicate_field", self.store().snapshot)

    def test_unsupported_schema_and_malformed_json_are_rejected(self):
        document = self.initialized_document()
        document["schema_version"] = 2
        self.assert_document_error("state_schema_unsupported", document)
        self.state_path.write_text("{" + PRIVATE_SENTINEL, encoding="utf-8")
        self.assert_error("state_json_invalid", self.store().snapshot)

    def test_invalid_utf8_and_non_object_state_are_rejected(self):
        self.store().initialize_or_load()
        self.state_path.write_bytes(b"\xff\xfe")
        self.assert_error("state_encoding_invalid", self.store().snapshot)
        self.state_path.write_text("[]", encoding="utf-8")
        self.assert_error("state_document_invalid", self.store().snapshot)

    def test_missing_state_fails_without_implicit_initialization(self):
        self.assert_error("state_not_initialized", self.store().snapshot)
        self.assertFalse(self.state_path.exists())

    def test_invalid_top_level_field_types_are_rejected(self):
        valid = self.initialized_document()
        cases = (
            ("registry_fingerprint", 7, "state_fingerprint_invalid"),
            ("registry_fingerprint", "A" * 64, "state_fingerprint_invalid"),
            ("cycle_number", 0, "state_cycle_invalid"),
            ("cycle_number", True, "state_cycle_invalid"),
            ("cycle_order", "BL-001-v1", "state_cycle_order_invalid"),
            ("last_consumed_id", 1, "state_last_consumed_invalid"),
            ("reservation", [], "state_reservation_invalid"),
        )
        for field, value, code in cases:
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(valid)
                document[field] = value
                self.assert_document_error(code, document)

    def test_malformed_cycle_team_id_is_rejected(self):
        document = self.initialized_document()
        document["cycle_order"][0] = PRIVATE_SENTINEL
        self.assert_document_error("state_cycle_order_invalid", document)

    def test_duplicate_cycle_ids_are_rejected(self):
        document = self.initialized_document()
        document["cycle_order"][1] = document["cycle_order"][0]
        self.assert_document_error("state_cycle_order_duplicate", document)

    def test_missing_active_and_inactive_or_unknown_ids_are_rejected(self):
        valid = self.initialized_document()
        document = copy.deepcopy(valid)
        document["cycle_order"] = document["cycle_order"][:-1]
        self.assert_document_error("state_cycle_order_mismatch", document)
        for replacement in ("BL-999-v1", "BL-777-v1"):
            with self.subTest(replacement=replacement):
                document = copy.deepcopy(valid)
                document["cycle_order"][0] = replacement
                self.assert_document_error("state_cycle_order_mismatch", document)

    def test_invalid_next_index_is_rejected(self):
        valid = self.initialized_document()
        for value in (-1, len(self.active_ids) + 1, True):
            with self.subTest(value=value):
                document = copy.deepcopy(valid)
                document["next_index"] = value
                self.assert_document_error("state_next_index_invalid", document)

    def test_invalid_last_consumed_id_is_rejected(self):
        valid = self.initialized_document()
        for value in ("BL-777-v1", self.active_ids[0]):
            with self.subTest(value=value):
                document = copy.deepcopy(valid)
                document["last_consumed_id"] = value
                self.assert_document_error("state_last_consumed_invalid", document)

    def test_malformed_reservation_id_and_phase_are_rejected(self):
        store = self.store()
        store.initialize_or_load()
        store.reserve_next()
        valid = self.state_document()
        for field, value, code in (
            ("reservation_id", PRIVATE_SENTINEL, "reservation_id_invalid"),
            ("phase", "arbitrary", "reservation_phase_invalid"),
        ):
            with self.subTest(field=field):
                document = copy.deepcopy(valid)
                document["reservation"][field] = value
                self.assert_document_error(code, document)

    def test_reservation_requires_exact_fields(self):
        store = self.store()
        store.initialize_or_load()
        store.reserve_next()
        valid = self.state_document()
        missing = copy.deepcopy(valid)
        del missing["reservation"]["position"]
        self.assert_document_error("state_missing_required_field", missing)
        unexpected = copy.deepcopy(valid)
        unexpected["reservation"]["private"] = PRIVATE_SENTINEL
        self.assert_document_error("state_unexpected_field", unexpected)

    def test_reservation_cycle_team_and_position_mismatches_are_rejected(self):
        store = self.store()
        store.initialize_or_load()
        store.reserve_next()
        valid = self.state_document()
        cases = (
            ("cycle_number", 2),
            ("team_id", self.active_ids[1]),
            ("position", 1),
        )
        for field, value in cases:
            with self.subTest(field=field):
                document = copy.deepcopy(valid)
                document["reservation"][field] = value
                self.assert_document_error("state_reservation_invalid", document)

    def test_reservation_at_completed_cycle_is_rejected(self):
        store = self.store()
        store.initialize_or_load()
        for _ in self.active_ids:
            self.consume(store)
        document = self.state_document()
        document["reservation"] = {
            "reservation_id": "1" * 32,
            "team_id": self.active_ids[-1],
            "cycle_number": 1,
            "position": len(self.active_ids),
            "phase": "reserved",
        }
        self.assert_document_error("state_reservation_invalid", document)

    def test_reservation_cannot_repeat_last_consumed_team(self):
        store = self.store(random_source=BoundaryCollisionRandom())
        store.initialize_or_load()
        for _ in self.active_ids:
            self.consume(store)
        store.reserve_next()
        document = self.state_document()
        document["last_consumed_id"] = document["reservation"]["team_id"]
        self.assert_document_error("state_cycle_boundary_repeat", document)

    def test_loaded_state_rejects_cycle_boundary_repeat_before_reservation(self):
        store = self.store(random_source=BoundaryCollisionRandom())
        store.initialize_or_load()
        for _ in self.active_ids:
            self.consume(store)
        reservation = store.reserve_next()
        store.release_reservation(reservation.reservation_id)
        document = self.state_document()
        previous_last = document["last_consumed_id"]
        previous_position = document["cycle_order"].index(previous_last)
        document["cycle_order"][0], document["cycle_order"][previous_position] = (
            document["cycle_order"][previous_position],
            document["cycle_order"][0],
        )
        self.assert_document_error("state_cycle_boundary_repeat", document)

    def test_semantic_registry_change_causes_fingerprint_mismatch(self):
        self.store().initialize_or_load()
        self._write_registry(registry_version="3.1")
        changed_registry = self.reload_registry()
        self.assert_error(
            "registry_fingerprint_mismatch",
            self.store(registry=changed_registry).snapshot,
        )

    def test_json_formatting_and_key_order_do_not_change_fingerprint(self):
        original = compute_registry_fingerprint(self.registry)
        document = self._registry_document()
        reordered = {
            "entries": [
                dict(reversed(tuple(entry.items())))
                for entry in document["entries"]
            ],
            "format_id": document["format_id"],
            "registry_version": document["registry_version"],
            "schema_version": document["schema_version"],
        }
        self.registry_path.write_text(json.dumps(reordered, indent=4), encoding="utf-8")
        reloaded = self.reload_registry()
        self.assertEqual(original, compute_registry_fingerprint(reloaded))

    def test_registry_entry_order_does_not_change_fingerprint(self):
        reordered = replace(
            self.registry,
            entries=tuple(reversed(self.registry.entries)),
        )
        self.assertEqual(
            compute_registry_fingerprint(self.registry),
            compute_registry_fingerprint(reordered),
        )

    def test_every_registry_semantic_field_affects_fingerprint(self):
        original = compute_registry_fingerprint(self.registry)
        first = self.registry.entries[0]
        changed_entries = (
            replace(first, team_id="BL-123-v1"),
            replace(first, active=False),
            replace(first, relative_team_path="teams/changed.team"),
            replace(first, sha256="0" * 64),
        )
        changed_registries = (
            replace(self.registry, schema_version=2),
            replace(self.registry, registry_version="3.1"),
            replace(self.registry, format_id="gen9nationaldex"),
            *(
                replace(
                    self.registry,
                    entries=(changed_entry, *self.registry.entries[1:]),
                )
                for changed_entry in changed_entries
            ),
        )
        for changed in changed_registries:
            with self.subTest(changed=repr(changed)):
                self.assertNotEqual(original, compute_registry_fingerprint(changed))

    def test_absolute_path_and_current_file_contents_do_not_affect_fingerprint(self):
        original = compute_registry_fingerprint(self.registry)
        first = self.registry.entries[0]
        relocated = replace(
            self.registry,
            entries=(
                replace(first, resolved_team_path=self.base / "different.team"),
                *self.registry.entries[1:],
            ),
        )
        first.resolved_team_path.write_bytes(b"changed-synthetic-contents\n")
        self.assertEqual(original, compute_registry_fingerprint(relocated))
        self.assertEqual(original, compute_registry_fingerprint(self.registry))


class TestBlindPoolBagAtomicity(BlindPoolBagFixture):
    def assert_failed_write_preserves_state(
        self,
        target: str,
        side_effect: BaseException,
    ):
        store = self.store()
        store.initialize_or_load()
        before = self.state_path.read_bytes()
        with mock.patch(target, side_effect=side_effect):
            self.assert_error("atomic_state_write_failed", store.reserve_next)
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual([], list(self.state_directory.glob("*.tmp")))
        self.assertEqual([], list(self.state_directory.glob(".*.tmp")))

    def assert_stream_failure_preserves_state(self, failure_point: str) -> None:
        store = self.store()
        store.initialize_or_load()
        before = self.state_path.read_bytes()
        original_fdopen = os.fdopen

        def failing_fdopen(*args, **kwargs):
            return FailingTextStream(
                original_fdopen(*args, **kwargs),
                failure_point,
            )

        with mock.patch(
            "fp.data.blind_pool.state.os.fdopen",
            side_effect=failing_fdopen,
        ):
            self.assert_error("atomic_state_write_failed", store.reserve_next)
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual([], list(self.state_directory.glob("*.tmp")))
        self.assertEqual([], list(self.state_directory.glob(".*.tmp")))

    def test_serialization_failure_preserves_previous_state(self):
        self.assert_failed_write_preserves_state(
            "fp.data.blind_pool.state._serialize_state",
            TypeError("synthetic serialization failure"),
        )

    def test_write_failure_preserves_previous_state(self):
        self.assert_stream_failure_preserves_state("write")

    def test_flush_failure_preserves_previous_state(self):
        self.assert_stream_failure_preserves_state("flush")

    def test_file_fsync_failure_preserves_previous_state(self):
        self.assert_failed_write_preserves_state(
            "fp.data.blind_pool.state.os.fsync",
            OSError("synthetic fsync failure"),
        )

    def test_atomic_replace_failure_preserves_previous_state(self):
        self.assert_failed_write_preserves_state(
            "fp.data.blind_pool.state.os.replace",
            OSError("synthetic replace failure"),
        )

    def test_successful_writes_leave_no_temporary_residue(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        store.commit_reservation(reservation.reservation_id)
        self.assertEqual([], list(self.state_directory.glob("*.tmp")))
        self.assertEqual([], list(self.state_directory.glob(".*.tmp")))

    def test_directory_fsync_ignores_only_unsupported_operation_errors(self):
        unsupported = OSError(errno.EINVAL, "unsupported")
        with mock.patch("fp.data.blind_pool.state.os.name", "posix"), mock.patch(
            "fp.data.blind_pool.state.os.open",
            side_effect=unsupported,
        ):
            _fsync_directory(self.state_directory)

        unexpected = OSError(errno.EIO, PRIVATE_SENTINEL)
        with mock.patch("fp.data.blind_pool.state.os.name", "posix"), mock.patch(
            "fp.data.blind_pool.state.os.open",
            side_effect=unexpected,
        ), self.assertRaises(OSError) as captured:
            _fsync_directory(self.state_directory)
        self.assertIs(unexpected, captured.exception)


class TestBlindPoolBagLocking(BlindPoolBagFixture):
    def test_lexical_aliases_share_one_canonical_in_process_mutex(self):
        alias_directory = self.state_directory / "alias"
        alias_directory.mkdir()
        canonical = BlindPoolStateLock(self.state_config.lock_path)
        alias = BlindPoolStateLock(
            alias_directory / ".." / self.state_config.lock_path.name
        )
        self.assertIs(canonical._mutex, alias._mutex)

    def test_lock_timeout_is_sanitized(self):
        store = self.store(lock_timeout_seconds=0.05)
        store.initialize_or_load()
        with BlindPoolStateLock(self.state_config.lock_path, timeout_seconds=1.0):
            error = self.assert_error("state_lock_timeout", store.snapshot)
        self.assertNotIn(str(self.state_config.lock_path), str(error))

    def test_two_store_instances_cannot_reserve_same_position(self):
        first_store = self.store(reservation_ids=["1" * 32])
        second_store = self.store(reservation_ids=["2" * 32])
        first_store.initialize_or_load()
        first = first_store.reserve_next()
        self.assert_error("unresolved_reservation_exists", second_store.reserve_next)
        self.assertEqual(first, second_store.snapshot().reservation)

    def test_concurrent_threads_serialize_reserve(self):
        self.store().initialize_or_load()
        barrier = threading.Barrier(2)
        results: list[tuple[str, str]] = []
        results_lock = threading.Lock()

        def reserve(identifier: str) -> None:
            store = self.store(reservation_ids=[identifier])
            barrier.wait(timeout=2)
            try:
                outcome = ("reserved", store.reserve_next().team_id)
            except BlindPoolValidationError as error:
                outcome = ("error", error.code)
            with results_lock:
                results.append(outcome)

        threads = [
            threading.Thread(target=reserve, args=("a" * 32,)),
            threading.Thread(target=reserve, args=("b" * 32,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(1, sum(result[0] == "reserved" for result in results))
        self.assertEqual(
            [("error", "unresolved_reservation_exists")],
            [result for result in results if result[0] == "error"],
        )

    def test_simultaneous_initialization_produces_one_valid_state(self):
        barrier = threading.Barrier(2)
        states = []

        def initialize(source) -> None:
            store = self.store(random_source=source)
            barrier.wait(timeout=2)
            states.append(store.initialize_or_load())

        threads = [
            threading.Thread(target=initialize, args=(IdentityRandom(),)),
            threading.Thread(target=initialize, args=(BoundaryCollisionRandom(),)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(2, len(states))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0], self.store().snapshot())

    def test_cross_process_reservations_are_serialized(self):
        self.store().initialize_or_load()
        code = (
            "import os; "
            "from fp.data.blind_pool import *; "
            "c=load_blind_pool_config(os.environ); "
            "s=load_blind_pool_state_config(os.environ); "
            "r=load_blind_pool_registry(c); "
            "b=BlindPoolBagStore(s,r,lock_timeout_seconds=2); "
            "\ntry:\n x=b.reserve_next(); print('reserved')"
            "\nexcept BlindPoolValidationError as e:\n print(e.code)"
        )
        environment = dict(os.environ, **self.environment)
        processes = []
        for _ in range(2):
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", code],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            self.addCleanup(_stop_process, process)
        outputs = []
        for process in processes:
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                _stop_process(process)
                self.fail("cross-process lock test timed out")
            self.assertEqual(0, process.returncode, stderr)
            outputs.append(stdout.strip())
        self.assertCountEqual(["reserved", "unresolved_reservation_exists"], outputs)


class TestBlindPoolBagPrivacy(BlindPoolBagFixture):
    def test_logs_exceptions_tracebacks_and_reprs_are_sanitized(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        package_logger = logging.getLogger("fp.data.blind_pool")
        old_level = package_logger.level
        package_logger.setLevel(logging.DEBUG)
        package_logger.addHandler(handler)
        try:
            store = self.store()
            state = store.initialize_or_load()
            reservation = store.reserve_next()
            store.release_reservation(reservation.reservation_id)
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(old_level)
        raw_state = self.state_path.read_text(encoding="utf-8")
        fingerprint = state.registry_fingerprint
        digest = self.registry.entries[0].sha256
        rendered = output.getvalue() + repr(store) + repr(state) + repr(reservation)
        for forbidden in (
            PRIVATE_SENTINEL,
            str(self.private_root),
            str(self.registry_path),
            str(self.state_path),
            digest,
            fingerprint,
            raw_state,
            reservation.reservation_id,
        ):
            self.assertNotIn(forbidden, rendered)

        self.state_path.write_text("{" + PRIVATE_SENTINEL, encoding="utf-8")
        with self.assertRaises(BlindPoolValidationError) as captured:
            store.snapshot()
        formatted = "".join(
            traceback.format_exception(
                type(captured.exception),
                captured.exception,
                captured.exception.__traceback__,
            )
        )
        self.assertNotIn(PRIVATE_SENTINEL, formatted)
        self.assertNotIn(str(self.state_path), formatted)
        self.assertIsNone(captured.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
