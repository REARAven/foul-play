from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fp.data.blind_pool import (
    BlindChallengeToken,
    BlindPoolBagStore,
    BlindPoolConfig,
    BlindPoolStateConfig,
    BlindPoolValidationError,
    BlindTeamPublicIdentity,
    LEGACY_STATE_SCHEMA_VERSION,
    PUBLIC_TEAM_KIND_PLAYER,
    STATE_SCHEMA_VERSION,
    load_blind_pool_registry,
    migrate_blind_pool_state_schema_2_to_3,
    validate_blind_pool_config,
    validate_blind_pool_state_config,
)


ROOT = Path(__file__).resolve().parents[1]
TOKEN = BlindChallengeToken("a" * 32)
PLAYER_ID = "player-team:" + "b" * 32
PLAYER = BlindTeamPublicIdentity(PLAYER_ID, "Arc-H HO", PUBLIC_TEAM_KIND_PLAYER)


class IdentityRandom:
    def shuffle(self, values):
        return None

    def randrange(self, start, stop=None):
        return start


class TeamSelectionStateFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        teams = self.private / "teams"
        state_dir = self.private / "state"
        teams.mkdir(parents=True)
        state_dir.mkdir()
        self.registry_path = self.private / "registry.json"
        entries = []
        for team_id in ("BL-001-v1", "BL-002-v1", "BL-003-v1"):
            payload = ("synthetic-" + team_id).encode()
            path = teams / (team_id + ".team")
            path.write_bytes(payload)
            entries.append(
                {
                    "team_id": team_id,
                    "active": True,
                    "team_file": "teams/" + path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        self.registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_version": "1.0",
                    "format_id": "gen9tugs",
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )
        self.state_path = state_dir / "selection.json"
        pool = validate_blind_pool_config(
            BlindPoolConfig(self.private, self.registry_path),
            repository_root=ROOT,
        )
        self.config = validate_blind_pool_state_config(
            BlindPoolStateConfig(pool, self.state_path), repository_root=ROOT
        )
        self.registry = load_blind_pool_registry(pool)

    def store(self, *, team_mode=True):
        return BlindPoolBagStore(
            self.config,
            self.registry,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: "c" * 32,
            team_mode=team_mode,
        )

    def error_code(self, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        return caught.exception.code


class TeamSelectionStateTests(TeamSelectionStateFixture):
    def test_team_store_initializes_strict_schema_three_idle_state(self):
        state = self.store().initialize_or_load()
        self.assertEqual(3, STATE_SCHEMA_VERSION)
        self.assertEqual(STATE_SCHEMA_VERSION, state.schema_version)
        self.assertIsNone(state.reservation)

    def test_team_reservation_requires_token_and_player_identity(self):
        store = self.store()
        store.initialize_or_load()
        self.assertEqual(
            "challenge_token_required", self.error_code(store.reserve_next)
        )
        self.assertEqual(
            "player_team_identity_required",
            self.error_code(lambda: store.reserve_next(TOKEN)),
        )

    def test_team_reservation_persists_every_authoritative_identity(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next(TOKEN, PLAYER)
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))["reservation"]
        self.assertEqual(PLAYER_ID, raw["player_team_id"])
        self.assertEqual("Arc-H HO", raw["player_team_display_name"])
        self.assertEqual("a" * 32, raw["challenge_token"])
        self.assertEqual("BL-001-v1", raw["team_id"])
        self.assertNotIn(PLAYER_ID, repr(reservation))
        self.assertNotIn("BL-001-v1", repr(reservation))

    def test_mark_accept_sent_preserves_team_identity(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next(TOKEN, PLAYER)
        state = store.mark_accept_sent(reservation.reservation_id, TOKEN)
        self.assertEqual(PLAYER_ID, state.reservation.player_team_id)
        self.assertEqual("Arc-H HO", state.reservation.player_team_display_name)

    def test_release_clears_complete_identity_without_consumption(self):
        store = self.store()
        before = store.initialize_or_load()
        reservation = store.reserve_next(TOKEN, PLAYER)
        after = store.release_reservation(reservation.reservation_id)
        self.assertIsNone(after.reservation)
        self.assertEqual(before.next_index, after.next_index)

    def test_room_commit_consumes_schema_three_reservation(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next(TOKEN, PLAYER)
        store.mark_accept_sent(reservation.reservation_id, TOKEN)
        after = store.commit_room_created(reservation.reservation_id)
        self.assertIsNone(after.reservation)
        self.assertEqual(1, after.next_index)
        self.assertEqual("BL-001-v1", after.last_consumed_id)

    def test_legacy_state_does_not_parse_in_team_mode(self):
        legacy = self.store(team_mode=False)
        state = legacy.initialize_or_load()
        self.assertEqual(LEGACY_STATE_SCHEMA_VERSION, state.schema_version)
        self.assertEqual(
            "state_schema_unsupported",
            self.error_code(self.store().snapshot),
        )

    def test_explicit_idle_migration_changes_only_schema(self):
        legacy = self.store(team_mode=False)
        legacy.initialize_or_load()
        first = legacy.reserve_next()
        legacy.commit_reservation(first.reservation_id)
        before = legacy.snapshot()
        before_raw = self.state_path.read_bytes()
        migrated = migrate_blind_pool_state_schema_2_to_3(self.config, self.registry)
        expected_raw = before_raw.replace(
            b'"schema_version":2', b'"schema_version":3', 1
        )
        self.assertEqual(expected_raw, self.state_path.read_bytes())
        self.assertEqual(STATE_SCHEMA_VERSION, migrated.schema_version)
        self.assertEqual(before.registry_fingerprint, migrated.registry_fingerprint)
        self.assertEqual(before.cycle_number, migrated.cycle_number)
        self.assertEqual(before.cycle_order, migrated.cycle_order)
        self.assertEqual(before.next_index, migrated.next_index)
        self.assertEqual(before.last_consumed_id, migrated.last_consumed_id)

    def test_migration_refuses_any_legacy_reservation(self):
        legacy = self.store(team_mode=False)
        legacy.initialize_or_load()
        legacy.reserve_next(TOKEN)
        self.assertEqual(
            "state_schema_migration_reservation_unresolved",
            self.error_code(
                lambda: migrate_blind_pool_state_schema_2_to_3(
                    self.config, self.registry
                )
            ),
        )

    def test_consumed_team_never_reenters_after_migration(self):
        legacy = self.store(team_mode=False)
        legacy.initialize_or_load()
        consumed = legacy.reserve_next()
        legacy.commit_reservation(consumed.reservation_id)
        migrate_blind_pool_state_schema_2_to_3(self.config, self.registry)
        reservation = self.store().reserve_next(TOKEN, PLAYER)
        self.assertNotEqual(consumed.team_id, reservation.team_id)

    def test_team_state_rejects_identityless_handwritten_reservation(self):
        store = self.store()
        store.initialize_or_load()
        document = json.loads(self.state_path.read_text(encoding="utf-8"))
        document["reservation"] = {
            "reservation_id": "c" * 32,
            "team_id": document["cycle_order"][0],
            "cycle_number": 1,
            "position": 0,
            "phase": "reserved",
            "challenge_token": "a" * 32,
        }
        self.state_path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(
            "state_missing_required_field", self.error_code(store.snapshot)
        )

    def test_legacy_reservation_rejects_team_identity_fields(self):
        store = self.store(team_mode=False)
        store.initialize_or_load()
        self.assertEqual(
            "player_team_identity_unexpected",
            self.error_code(lambda: store.reserve_next(TOKEN, PLAYER)),
        )


if __name__ == "__main__":
    unittest.main()
