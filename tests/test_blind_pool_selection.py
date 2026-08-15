from dataclasses import asdict
import json
from pathlib import Path
import pickle
import unittest

from fp.data.blind_pool.bag import BlindPoolBagStore
from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.fingerprint import compute_registry_fingerprint
from fp.data.blind_pool.canonical_registry import load_canonical_runtime_registry
from fp.data.blind_pool.models import (
    BlindPoolConfig,
    BlindPoolEntry,
    BlindPoolRegistry,
    BlindPoolStateConfig,
)
from fp.data.blind_pool.selection import (
    BlindPoolSelectionSnapshot,
    create_canonical_selection_snapshot,
    create_raw_selection_snapshot,
)
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture
from tests.test_blind_pool_lifecycle import IdentityRandom


class SelectionFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id, active in (
            ("BL-020-v1", True),
            ("BL-003-v1", False),
            ("BL-001-v1", True),
        ):
            self.fixture.add_artifact(team_id, active=active)
        self.canonical_registry = self.fixture.load()
        self.state_directory = self.fixture.private_root / "state"
        self.state_directory.mkdir()
        self.state_path = self.state_directory / "bag.json"
        self.pool_config = BlindPoolConfig(
            self.fixture.private_root.resolve(),
            self.fixture.registry_path.resolve(),
        )
        self.state_config = BlindPoolStateConfig(
            self.pool_config,
            self.state_path.resolve(),
        )

    def raw_registry(self, active_ids=("BL-001-v1", "BL-020-v1")):
        raw_directory = self.fixture.private_root / "raw"
        raw_directory.mkdir(exist_ok=True)
        entries = []
        for team_id in ("BL-001-v1", "BL-003-v1", "BL-020-v1"):
            path = raw_directory / (team_id + ".team")
            path.write_text("synthetic", encoding="utf-8")
            entries.append(
                BlindPoolEntry(
                    team_id=team_id,
                    active=team_id in active_ids,
                    relative_team_path="raw/" + path.name,
                    resolved_team_path=path.resolve(),
                    sha256="1" * 64,
                )
            )
        return BlindPoolRegistry(1, "3.0", "gen9tugs", tuple(entries))

    def canonical_store(self, registry=None, state_config=None):
        identifiers = iter(format(index, "032x") for index in range(1, 100))
        return BlindPoolBagStore.from_canonical_registry(
            state_config or self.state_config,
            registry or self.canonical_registry,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(identifiers),
            lock_timeout_seconds=1,
        )

    def assert_code(self, code, operation):
        with self.assertRaises(BlindPoolValidationError) as caught:
            operation()
        self.assertEqual(code, caught.exception.code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        return caught.exception


class SelectionSnapshotTests(SelectionFixture):
    def test_raw_adapter_preserves_active_order_and_historical_fingerprint(self):
        registry = self.raw_registry()
        snapshot = create_raw_selection_snapshot(
            registry,
            registry_path=self.pool_config.registry_path,
        )
        self.assertEqual(
            tuple(entry.team_id for entry in registry.active_entries),
            snapshot.active_ids,
        )
        self.assertEqual(
            compute_registry_fingerprint(registry), snapshot.registry_fingerprint
        )

    def test_canonical_adapter_uses_registry_identity_directly(self):
        snapshot = create_canonical_selection_snapshot(self.canonical_registry)
        self.assertEqual(("BL-001-v1", "BL-020-v1"), snapshot.active_ids)
        self.assertIs(
            snapshot.registry_fingerprint,
            self.canonical_registry.registry_fingerprint,
        )

    def test_snapshot_is_deeply_immutable_and_representation_is_safe(self):
        snapshot = create_canonical_selection_snapshot(self.canonical_registry)
        with self.assertRaises(AttributeError):
            snapshot.active_ids = ()
        with self.assertRaises(TypeError):
            snapshot.active_ids[0] = "BL-999-v1"
        rendered = repr(snapshot)
        self.assertNotIn(str(self.fixture.private_root), rendered)
        self.assertNotIn(snapshot.registry_fingerprint, rendered)
        self.assertNotIn("syntheticspecies", rendered)
        self.assertNotIn("packed", rendered.casefold())

    def test_snapshot_cannot_serialize_private_collision_identities(self):
        snapshot = create_canonical_selection_snapshot(self.canonical_registry)
        with self.assertRaises(TypeError):
            pickle.dumps(snapshot)
        with self.assertRaises(TypeError):
            asdict(snapshot)

    def test_snapshot_contains_no_team_artifact_or_sidecar_surface(self):
        snapshot = create_canonical_selection_snapshot(self.canonical_registry)
        self.assertIsInstance(snapshot, BlindPoolSelectionSnapshot)
        for name in (
            "packed_for_submission",
            "new_battle_team_projection",
            "metadata",
            "sidecar",
            "entries",
        ):
            self.assertFalse(hasattr(snapshot, name))

    def test_raw_parent_fingerprint_vector_remains_covered(self):
        registry = BlindPoolRegistry(
            schema_version=1,
            registry_version="7.4",
            format_id="gen9tugs",
            entries=(
                BlindPoolEntry(
                    team_id="BL-010-v2",
                    active=False,
                    relative_team_path="nested/ten.team",
                    resolved_team_path=Path("C:/synthetic/ten.team"),
                    sha256="a" * 64,
                ),
                BlindPoolEntry(
                    team_id="BL-002-v1",
                    active=True,
                    relative_team_path="two.team",
                    resolved_team_path=Path("D:/synthetic/two.team"),
                    sha256="b" * 64,
                ),
            ),
        )
        snapshot = create_raw_selection_snapshot(registry)
        self.assertEqual(
            compute_registry_fingerprint(registry), snapshot.registry_fingerprint
        )


class CanonicalBagTests(SelectionFixture):
    def test_canonical_bag_preserves_reserve_commit_and_release_semantics(self):
        store = self.canonical_store()
        initial = store.initialize_or_load()
        first = store.reserve_next()
        self.assertEqual(initial.cycle_order[0], first.team_id)
        self.assertEqual(0, store.snapshot().next_index)
        store.release_reservation(first.reservation_id)
        repeated = store.reserve_next()
        self.assertEqual(first.team_id, repeated.team_id)
        committed = store.commit_reservation(repeated.reservation_id)
        self.assertEqual(1, committed.next_index)
        self.assertIsNone(committed.reservation)
        before = self.state_path.read_bytes()
        self.assert_code(
            "reservation_not_found",
            lambda: store.commit_reservation(repeated.reservation_id),
        )
        self.assertEqual(before, self.state_path.read_bytes())

    def test_canonical_bag_restart_and_unresolved_reservation_behavior(self):
        store = self.canonical_store()
        store.initialize_or_load()
        reservation = store.reserve_next()
        restarted = self.canonical_store()
        self.assertEqual(reservation, restarted.snapshot().reservation)
        self.assert_code("unresolved_reservation_exists", restarted.reserve_next)
        restarted.release_reservation(reservation.reservation_id)

    def test_canonical_bag_cycle_rollover_has_no_boundary_repeat(self):
        store = self.canonical_store()
        store.initialize_or_load()
        consumed = []
        for _ in range(3):
            reservation = store.reserve_next()
            consumed.append(reservation.team_id)
            store.commit_reservation(reservation.reservation_id)
        self.assertEqual(3, len(consumed))
        self.assertNotEqual(consumed[1], consumed[2])
        self.assertEqual(set(consumed[:2]), set(self.canonical_registry.active_ids))

    def test_canonical_fingerprint_change_rejects_stale_state_without_rewrite(self):
        store = self.canonical_store()
        store.initialize_or_load()
        before = self.state_path.read_bytes()
        self.fixture.write_registry(
            document={
                "schema_version": 1,
                "registry_version": "1.0.1",
                "format_id": "gen9tugs",
                "artifact_schema_version": 1,
                "metadata_schema_version": 1,
                "entries": self.fixture.entries,
            }
        )
        changed = self.fixture.load()
        self.assert_code(
            "registry_fingerprint_mismatch",
            self.canonical_store(changed).snapshot,
        )
        self.assertEqual(before, self.state_path.read_bytes())

    def test_raw_state_rejected_by_canonical_selection_without_rewrite(self):
        raw = self.raw_registry()
        identifiers = iter(format(index, "032x") for index in range(1, 100))
        raw_store = BlindPoolBagStore(
            self.state_config,
            raw,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(identifiers),
            lock_timeout_seconds=1,
        )
        raw_store.initialize_or_load()
        before = self.state_path.read_bytes()
        self.assert_code(
            "registry_fingerprint_mismatch",
            self.canonical_store().snapshot,
        )
        self.assertEqual(before, self.state_path.read_bytes())

    def test_existing_raw_state_loads_through_generalized_store_unchanged(self):
        raw = self.raw_registry()
        first = BlindPoolBagStore(
            self.state_config,
            raw,
            random_source=IdentityRandom(),
            lock_timeout_seconds=1,
        )
        expected = first.initialize_or_load()
        before = self.state_path.read_bytes()
        second = BlindPoolBagStore(
            self.state_config,
            raw,
            random_source=IdentityRandom(),
            lock_timeout_seconds=1,
        )
        self.assertEqual(expected, second.initialize_or_load())
        self.assertEqual(before, self.state_path.read_bytes())

    def test_raw_initial_state_changes_only_to_authorized_schema_v2(self):
        raw = self.raw_registry()
        store = BlindPoolBagStore(
            self.state_config,
            raw,
            random_source=IdentityRandom(),
            lock_timeout_seconds=1,
        )
        store.initialize_or_load()
        self.assertEqual(
            {
                "schema_version": 2,
                "registry_fingerprint": compute_registry_fingerprint(raw),
                "cycle_number": 1,
                "cycle_order": ["BL-001-v1", "BL-020-v1"],
                "next_index": 0,
                "last_consumed_id": None,
                "reservation": None,
            },
            json.loads(self.state_path.read_text(encoding="utf-8")),
        )

    def test_canonical_bag_requires_two_active_entries(self):
        single = SyntheticCanonicalFixture()
        self.addCleanup(single.close)
        single.add_artifact("BL-001-v1")
        registry = single.load()
        state_directory = single.private_root / "state"
        state_directory.mkdir()
        config = BlindPoolStateConfig(
            BlindPoolConfig(
                single.private_root.resolve(),
                single.registry_path.resolve(),
            ),
            (state_directory / "bag.json").resolve(),
        )
        self.assert_code(
            "insufficient_active_entries",
            lambda: BlindPoolBagStore.from_canonical_registry(config, registry),
        )


class CanonicalCollisionTests(SelectionFixture):
    def collision_config(self, state_path):
        return BlindPoolStateConfig(self.pool_config, Path(state_path).absolute())

    def test_state_rejects_canonical_root_schema_artifact_and_final_files(self):
        paths = (
            self.fixture.private_root / "canonical" / "state.json",
            self.fixture.schema_root / "state.json",
            self.fixture.schema_root / "BL-001-v1" / "state.json",
            self.fixture.schema_root / "BL-001-v1" / "packed.txt",
            self.fixture.schema_root / "BL-001-v1" / "team.json",
            self.fixture.schema_root / "BL-001-v1" / "metadata.json",
        )
        for path in paths:
            with self.subTest(name=path.name):
                self.assert_code(
                    "state_canonical_artifact_collision",
                    lambda path=path: self.canonical_store(
                        state_config=self.collision_config(path)
                    ),
                )

    def test_resolved_lexical_alias_into_canonical_tree_is_rejected(self):
        alias = self.fixture.schema_root / ".." / "schema-1" / "state.json"
        self.assert_code(
            "state_canonical_artifact_collision",
            lambda: self.canonical_store(state_config=self.collision_config(alias)),
        )

    def test_private_collision_check_rejects_lock_under_canonical_tree(self):
        selection = create_canonical_selection_snapshot(self.canonical_registry)
        outside = self.state_directory / "outside.json"
        lock = self.fixture.schema_root / "state.lock"
        self.assert_code(
            "state_lock_canonical_artifact_collision",
            lambda: selection._validate_collision_paths(
                configured_registry_path=self.pool_config.registry_path,
                state_path=outside.resolve(),
                lock_path=lock.resolve(),
            ),
        )

    def test_registry_identity_mismatch_fails_closed(self):
        other_registry = self.fixture.base / "other-registry.json"
        other_registry.write_text(json.dumps({}), encoding="utf-8")
        other_config = BlindPoolStateConfig(
            BlindPoolConfig(
                self.fixture.private_root.resolve(), other_registry.resolve()
            ),
            self.state_path.resolve(),
        )
        self.assert_code(
            "selection_registry_path_mismatch",
            lambda: self.canonical_store(state_config=other_config),
        )

    def test_state_and_lock_cannot_replace_canonical_registry(self):
        for registry_name, state_name, expected_code in (
            (
                "runtime-registry.json",
                "runtime-registry.json",
                "state_registry_collision",
            ),
            ("bag.json.lock", "bag.json", "state_lock_registry_collision"),
        ):
            with self.subTest(expected_code=expected_code):
                registry_path = self.fixture.private_root / registry_name
                registry_path.write_bytes(self.fixture.registry_path.read_bytes())
                registry = load_canonical_runtime_registry(
                    self.fixture.private_root,
                    registry_path,
                )
                config = BlindPoolStateConfig(
                    BlindPoolConfig(
                        self.fixture.private_root.resolve(),
                        registry_path.resolve(),
                    ),
                    (self.fixture.private_root / state_name).resolve(),
                )
                self.assert_code(
                    expected_code,
                    lambda: self.canonical_store(
                        registry=registry,
                        state_config=config,
                    ),
                )

    def test_symlink_alias_into_canonical_tree_is_rejected_when_supported(self):
        alias = self.fixture.private_root / "canonical-alias"
        try:
            alias.symlink_to(
                self.fixture.private_root / "canonical", target_is_directory=True
            )
        except OSError:
            self.skipTest("directory symlinks unavailable")
        self.assert_code(
            "state_canonical_artifact_collision",
            lambda: self.canonical_store(
                state_config=self.collision_config(alias / "state.json")
            ),
        )
