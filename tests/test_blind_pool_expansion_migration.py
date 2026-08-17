from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fp.data.blind_pool import (
    BlindCanonicalStartupConfig,
    BlindPoolBagState,
    BlindPoolConfig,
    BlindPoolExpansionMigrationResult,
    BlindPoolReservation,
    BlindPoolStateConfig,
    BlindPoolValidationError,
    create_canonical_selection_snapshot,
    load_canonical_runtime_registry,
    migrate_blind_canonical_pool_expansion,
    validate_blind_pool_bag_state,
)
from fp.data.blind_pool.maintenance import main as maintenance_main
from fp.data.blind_pool.state import (
    ACCEPT_SENT_PHASE,
    RESERVATION_PHASE,
    load_blind_pool_bag_state,
    write_blind_pool_bag_state_atomic,
)
from tests.test_blind_canonical_registry import (
    SyntheticCanonicalFixture,
    compact_json,
)


class IdentityRandom:
    def shuffle(self, values: list[str]) -> None:
        return None

    def randrange(self, start: int, stop: int | None = None) -> int:
        return start


class ReverseRandom:
    def shuffle(self, values: list[str]) -> None:
        values.reverse()

    def randrange(self, start: int, stop: int | None = None) -> int:
        return start


class ExpansionFixture:
    def __init__(self, old_count: int = 4, target_count: int = 7):
        self.source = SyntheticCanonicalFixture()
        self.target = SyntheticCanonicalFixture()
        self.old_ids = tuple(
            "BL-{:03d}-v1".format(index) for index in range(1, old_count + 1)
        )
        self.target_ids = tuple(
            "BL-{:03d}-v1".format(index) for index in range(1, target_count + 1)
        )
        for team_id in self.old_ids:
            self.source.add_artifact(team_id)
            self.target.add_artifact(team_id)
        for team_id in self.target_ids[old_count:]:
            self.target.add_artifact(team_id)
        self._write_target_registry()

        self.source_state_path = self.source.private_root / "runtime" / "state.json"
        self.target_state_path = self.target.private_root / "runtime" / "state.json"
        self.source_state_path.parent.mkdir()
        self.target_state_path.parent.mkdir()
        self.source_startup = BlindCanonicalStartupConfig(
            self.source.private_root,
            self.source.registry_path,
            self.source_state_path,
        )
        self.target_startup = BlindCanonicalStartupConfig(
            self.target.private_root,
            self.target.registry_path,
            self.target_state_path,
        )
        self.source_registry = self.source.load()
        self.source_selection = create_canonical_selection_snapshot(
            self.source_registry,
        )
        self.source_state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.source.private_root, self.source.registry_path),
            self.source_state_path,
        )
        self.write_source_state(next_index=2)

    def close(self) -> None:
        self.source.close()
        self.target.close()

    def _write_target_registry(self) -> None:
        self.target.write_registry(
            document={
                "schema_version": 1,
                "registry_version": "2.0.0",
                "format_id": "gen9tugs",
                "artifact_schema_version": 1,
                "metadata_schema_version": 1,
                "entries": self.target.entries,
            }
        )

    def write_source_state(
        self,
        *,
        next_index: int,
        cycle_number: int = 1,
        cycle_order: tuple[str, ...] | None = None,
        last_consumed_id: str | None = None,
        reservation_phase: str | None = None,
    ) -> BlindPoolBagState:
        order = self.old_ids if cycle_order is None else cycle_order
        if last_consumed_id is None and next_index:
            last_consumed_id = order[next_index - 1]
        reservation = None
        if reservation_phase is not None:
            reservation = BlindPoolReservation(
                reservation_id="a" * 32,
                team_id=order[next_index],
                cycle_number=cycle_number,
                position=next_index,
                phase=reservation_phase,
                challenge_token=None,
            )
        state = BlindPoolBagState(
            schema_version=2,
            registry_fingerprint=self.source_selection.registry_fingerprint,
            cycle_number=cycle_number,
            cycle_order=order,
            next_index=next_index,
            last_consumed_id=last_consumed_id,
            reservation=reservation,
        )
        write_blind_pool_bag_state_atomic(
            self.source_state_config,
            state,
            self.source_selection,
        )
        return state

    def migrate(self, *, random_source=None):
        return migrate_blind_canonical_pool_expansion(
            self.source_startup,
            self.target_startup,
            random_source=random_source or IdentityRandom(),
        )

    def target_state(self) -> BlindPoolBagState:
        registry = load_canonical_runtime_registry(
            self.target.private_root,
            self.target.registry_path,
        )
        selection = create_canonical_selection_snapshot(registry)
        config = BlindPoolStateConfig(
            BlindPoolConfig(self.target.private_root, self.target.registry_path),
            self.target_state_path,
        )
        return load_blind_pool_bag_state(config, selection)


class ExpansionMigrationTestCase(unittest.TestCase):
    def fixture(self, old_count: int = 4, target_count: int = 7) -> ExpansionFixture:
        fixture = ExpansionFixture(old_count, target_count)
        self.addCleanup(fixture.close)
        return fixture

    def assert_code(self, code: str, callback) -> BlindPoolValidationError:
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        return caught.exception

    def test_expansion_16_to_32_preserves_consumed_prefix(self):
        fixture = self.fixture(16, 32)
        source = fixture.write_source_state(next_index=7)
        fixture.migrate(random_source=ReverseRandom())
        migrated = fixture.target_state()
        self.assertEqual(source.cycle_order[:7], migrated.cycle_order[:7])

    def test_consumed_ids_do_not_recur_in_remaining_cycle(self):
        fixture = self.fixture()
        source = fixture.write_source_state(next_index=2)
        fixture.migrate()
        migrated = fixture.target_state()
        self.assertTrue(
            set(source.cycle_order[:2]).isdisjoint(migrated.cycle_order[2:])
        )

    def test_all_added_ids_appear_exactly_once(self):
        fixture = self.fixture()
        fixture.migrate()
        migrated = fixture.target_state()
        added = set(fixture.target_ids) - set(fixture.old_ids)
        self.assertEqual(added, set(migrated.cycle_order) & added)
        self.assertTrue(
            all(migrated.cycle_order.count(team_id) == 1 for team_id in added)
        )

    def test_all_old_unconsumed_ids_remain_exactly_once(self):
        fixture = self.fixture()
        source = fixture.write_source_state(next_index=2)
        fixture.migrate(random_source=ReverseRandom())
        migrated = fixture.target_state()
        for team_id in source.cycle_order[2:]:
            self.assertEqual(1, migrated.cycle_order[2:].count(team_id))

    def test_next_index_is_preserved(self):
        fixture = self.fixture()
        fixture.write_source_state(next_index=3)
        fixture.migrate()
        self.assertEqual(3, fixture.target_state().next_index)

    def test_cycle_number_is_preserved(self):
        fixture = self.fixture()
        fixture.write_source_state(next_index=2, cycle_number=5)
        fixture.migrate()
        self.assertEqual(5, fixture.target_state().cycle_number)

    def test_last_consumed_boundary_is_preserved(self):
        fixture = self.fixture()
        source = fixture.write_source_state(next_index=2, cycle_number=3)
        fixture.migrate()
        self.assertEqual(
            source.last_consumed_id, fixture.target_state().last_consumed_id
        )

    def test_completed_cycle_is_extended_only_with_additions(self):
        fixture = self.fixture()
        source = fixture.write_source_state(
            next_index=len(fixture.old_ids), cycle_number=2
        )
        fixture.migrate(random_source=ReverseRandom())
        migrated = fixture.target_state()
        self.assertEqual(source.cycle_order, migrated.cycle_order[: source.next_index])
        self.assertEqual(
            set(fixture.target_ids) - set(fixture.old_ids),
            set(migrated.cycle_order[source.next_index :]),
        )

    def test_later_cycle_boundary_cannot_immediately_repeat(self):
        fixture = self.fixture()
        boundary = fixture.old_ids[0]
        order = fixture.old_ids[1:] + (boundary,)
        fixture.write_source_state(
            next_index=0,
            cycle_number=4,
            cycle_order=order,
            last_consumed_id=boundary,
        )
        fixture.migrate(random_source=IdentityRandom())
        migrated = fixture.target_state()
        self.assertNotEqual(boundary, migrated.cycle_order[0])
        self.assertEqual(boundary, migrated.last_consumed_id)

    def test_migration_refuses_removal(self):
        fixture = self.fixture()
        fixture.target.entries[0]["active"] = False
        fixture._write_target_registry()
        self.assert_code("expansion_membership_invalid", fixture.migrate)

    def test_migration_refuses_changed_existing_registry_binding(self):
        fixture = self.fixture()
        team_id = fixture.old_ids[0]
        metadata_path = fixture.target.schema_root / team_id / "metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["source_sha256"] = "b" * 64
        raw = compact_json(metadata)
        metadata_path.write_bytes(raw)
        for entry in fixture.target.entries:
            if entry["team_id"] == team_id:
                entry["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        fixture._write_target_registry()
        self.assert_code("expansion_existing_binding_changed", fixture.migrate)

    def test_migration_refuses_changed_existing_artifact_file(self):
        fixture = self.fixture()
        artifact_path = fixture.target.schema_root / fixture.old_ids[0] / "packed.txt"
        artifact_path.write_bytes(b"synthetic changed payload")
        self.assert_code("expansion_existing_binding_changed", fixture.migrate)

    def test_migration_refuses_reserved_state(self):
        fixture = self.fixture()
        fixture.write_source_state(
            next_index=1,
            reservation_phase=RESERVATION_PHASE,
        )
        self.assert_code("expansion_reserved_unresolved", fixture.migrate)

    def test_migration_refuses_accept_sent_state(self):
        fixture = self.fixture()
        fixture.write_source_state(
            next_index=1,
            reservation_phase=ACCEPT_SENT_PHASE,
        )
        self.assert_code("expansion_accept_sent_unresolved", fixture.migrate)

    def test_migration_refuses_malformed_source_state(self):
        fixture = self.fixture()
        fixture.source_state_path.write_bytes(b"{malformed")
        self.assert_code("expansion_source_state_invalid", fixture.migrate)

    def test_migration_refuses_source_registry_fingerprint_mismatch(self):
        fixture = self.fixture()
        document = json.loads(fixture.source_state_path.read_bytes())
        document["registry_fingerprint"] = "f" * 64
        fixture.source_state_path.write_bytes(compact_json(document))
        self.assert_code("expansion_source_state_invalid", fixture.migrate)

    def test_migration_refuses_and_preserves_existing_target_state(self):
        fixture = self.fixture()
        sentinel = b"synthetic existing target state"
        fixture.target_state_path.write_bytes(sentinel)
        self.assert_code("expansion_target_state_exists", fixture.migrate)
        self.assertEqual(sentinel, fixture.target_state_path.read_bytes())

    def test_atomic_new_state_publication_cannot_clobber_target(self):
        fixture = self.fixture()
        target_registry = fixture.target.load()
        target_selection = create_canonical_selection_snapshot(target_registry)
        target_config = BlindPoolStateConfig(
            BlindPoolConfig(fixture.target.private_root, fixture.target.registry_path),
            fixture.target_state_path,
        )
        state = BlindPoolBagState(
            schema_version=2,
            registry_fingerprint=target_selection.registry_fingerprint,
            cycle_number=1,
            cycle_order=fixture.target_ids,
            next_index=0,
            last_consumed_id=None,
            reservation=None,
        )
        sentinel = b"synthetic race winner"
        fixture.target_state_path.write_bytes(sentinel)
        self.assert_code(
            "state_target_exists",
            lambda: write_blind_pool_bag_state_atomic(
                target_config,
                state,
                target_selection,
                replace_existing=False,
            ),
        )
        self.assertEqual(sentinel, fixture.target_state_path.read_bytes())

    def test_migrated_state_passes_normal_schema_v2_validation(self):
        fixture = self.fixture()
        result = fixture.migrate(random_source=ReverseRandom())
        state = fixture.target_state()
        target_registry = fixture.target.load()
        validated = validate_blind_pool_bag_state(
            {
                "schema_version": state.schema_version,
                "registry_fingerprint": state.registry_fingerprint,
                "cycle_number": state.cycle_number,
                "cycle_order": list(state.cycle_order),
                "next_index": state.next_index,
                "last_consumed_id": state.last_consumed_id,
                "reservation": None,
            },
            create_canonical_selection_snapshot(target_registry),
        )
        self.assertEqual(state, validated)
        self.assertEqual(2, state.schema_version)
        self.assertEqual(len(fixture.target_ids), result.target_active_count)

    def test_source_state_bytes_remain_unchanged(self):
        fixture = self.fixture()
        before = fixture.source_state_path.read_bytes()
        fixture.migrate()
        self.assertEqual(before, fixture.source_state_path.read_bytes())

    def test_default_random_source_is_system_random(self):
        fixture = self.fixture()
        with mock.patch("fp.data.blind_pool.migration.random.SystemRandom") as factory:
            factory.return_value = IdentityRandom()
            fixture.migrate(random_source=None)
        self.assertGreaterEqual(factory.call_count, 1)

    def test_cli_output_is_aggregate_only(self):
        result = BlindPoolExpansionMigrationResult(16, 32, 5, 27, "2.0.0")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "fp.data.blind_pool.maintenance.migrate_blind_canonical_pool_expansion",
            return_value=result,
        ), mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            code = maintenance_main(
                [
                    "migrate-expansion",
                    "--source-private-root",
                    str(Path(tempfile.gettempdir()) / "source-private"),
                    "--source-registry",
                    str(Path(tempfile.gettempdir()) / "source-registry.json"),
                    "--source-state",
                    str(Path(tempfile.gettempdir()) / "source-state.json"),
                    "--target-private-root",
                    str(Path(tempfile.gettempdir()) / "target-private"),
                    "--target-registry",
                    str(Path(tempfile.gettempdir()) / "target-registry.json"),
                    "--target-state",
                    str(Path(tempfile.gettempdir()) / "target-state.json"),
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertNotIn("BL-", stdout.getvalue())
        self.assertIn("source active: 16", stdout.getvalue())
        self.assertIn("target active: 32", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
