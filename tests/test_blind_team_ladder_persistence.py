from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from fp.data.blind_pool import (
    BlindPoolValidationError,
    BlindTeamPublicIdentity,
    BlindTeamPublicRegistryStore,
    BlindTeamRatingStateStore,
    BlindTeamResultLedgerStore,
    PUBLIC_TEAM_KIND_PLAYER,
    TEAM_OUTCOME_A_LOSS,
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_NO_RESULT,
    TEAM_OUTCOME_TIE,
    build_leaderboard,
    derive_blind_team_battle_id,
    validate_team_public_registry_config,
    validate_team_rating_state_config,
    validate_team_result_ledger_config,
)


ROOT = Path(__file__).resolve().parents[1]
PLAYER_TEAM_A = "player-team:" + "1" * 32
PLAYER_TEAM_B = "player-team:" + "2" * 32
BOT_A = "BL-001-v1"
BOT_B = "BL-002-v1"
FINGERPRINT = "f" * 64


class ReverseRandom:
    def shuffle(self, values):
        values.reverse()


class IdentityRandom:
    def shuffle(self, values):
        return None


class Clock:
    def __init__(self):
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        value = self.value
        self.value += timedelta(seconds=1)
        return value


class TeamPersistenceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.private = root / "private"
        self.external = root / "external"
        self.private.mkdir()
        self.external.mkdir()
        self.canonical = self.private / "canonical.json"
        self.canonical.write_text("{}", encoding="utf-8")
        self.selection = self.external / "selection.json"
        self.selection.write_text("{}", encoding="utf-8")
        self.public_path = self.external / "team-public.json"
        self.result_path = self.external / "team-results.json"
        self.rating_path = self.external / "team-ratings.json"
        self.public_store = BlindTeamPublicRegistryStore(
            validate_team_public_registry_config(
                self.public_path,
                private_root=self.private,
                canonical_registry_path=self.canonical,
                selection_state_path=self.selection,
                result_ledger_path=self.result_path,
                rating_state_path=self.rating_path,
                repository_root=ROOT,
            )
        )
        self.result_store = BlindTeamResultLedgerStore(
            validate_team_result_ledger_config(
                self.result_path,
                private_root=self.private,
                registry_path=self.canonical,
                selection_state_path=self.selection,
                rating_state_path=self.rating_path,
                public_registry_path=self.public_path,
                repository_root=ROOT,
            ),
            clock=Clock(),
        )
        self.rating_store = BlindTeamRatingStateStore(
            validate_team_rating_state_config(
                self.rating_path,
                private_root=self.private,
                canonical_registry_path=self.canonical,
                selection_state_path=self.selection,
                result_ledger_path=self.result_path,
                public_registry_path=self.public_path,
                repository_root=ROOT,
            )
        )

    def initialize(self):
        public = self.public_store.initialize(
            (BOT_A, BOT_B), random_source=ReverseRandom()
        )
        ledger = self.result_store.initialize_empty()
        ratings = self.rating_store.initialize(ledger)
        return public, ledger, ratings

    def register_player(self, team_id=PLAYER_TEAM_A, name="Arc-H HO"):
        return self.public_store.register_player(
            BlindTeamPublicIdentity(team_id, name, PUBLIC_TEAM_KIND_PLAYER)
        )

    def pending(
        self,
        *,
        player_team_id=PLAYER_TEAM_A,
        bot_team_id=BOT_A,
        reservation="a" * 32,
        room="battle-gen9tugs-1",
        player_account="player",
    ):
        return self.result_store.create_pending_intent(
            player_account_id=player_account,
            bot_account_id="bot",
            player_team_id=player_team_id,
            bot_team_id=bot_team_id,
            reservation_id=reservation,
            room_id=room,
            format_id="gen9tugs",
            registry_fingerprint=FINGERPRINT,
        )

    def complete(self, outcome=TEAM_OUTCOME_A_WIN, **kwargs):
        pending = self.pending(**kwargs)
        self.result_store.mark_selection_committed(pending.battle_id)
        return self.result_store.finalize_terminal(pending.battle_id, outcome)

    def assert_code(self, code, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        return caught.exception


class TeamResultLedgerTests(TeamPersistenceFixture):
    def test_empty_initialize_is_strict_and_refuses_nonempty_reinitialize(self):
        _, ledger, _ = self.initialize()
        self.assertEqual((0, 0), (ledger.completed_count, ledger.pending_count))
        raw = self.result_path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.pending()
        self.assert_code(
            "team_result_ledger_not_empty", self.result_store.initialize_empty
        )

    def test_strict_schema_and_duplicate_fields_fail_closed(self):
        self.initialize()
        raw = self.result_path.read_bytes()
        document = json.loads(raw)
        document["unexpected"] = True
        self.result_path.write_text(json.dumps(document), encoding="utf-8")
        self.assert_code("team_result_fields_invalid", self.result_store.load)
        duplicate = raw.replace(
            b'"schema_version":1',
            b'"schema_version":1,"schema_version":1',
            1,
        )
        self.result_path.write_bytes(duplicate)
        self.assert_code("team_result_duplicate_field", self.result_store.load)

    def test_state_path_cannot_collide_with_selection_owner_lock(self):
        self.assert_code(
            "team_result_path_collision",
            lambda: validate_team_result_ledger_config(
                self.selection.with_name(self.selection.name + ".owner.lock"),
                private_root=self.private,
                registry_path=self.canonical,
                selection_state_path=self.selection,
                rating_state_path=self.rating_path,
                public_registry_path=self.public_path,
                repository_root=ROOT,
            ),
        )

    def test_pending_phases_and_private_identities_are_durable(self):
        self.initialize()
        pending = self.pending()
        self.assertEqual("room_validated", pending.phase)
        self.assertNotIn(PLAYER_TEAM_A, repr(pending))
        document = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.assertEqual(PLAYER_TEAM_A, document["pending_result"]["player_team_id"])
        self.assertEqual(BOT_A, document["pending_result"]["bot_team_id"])
        committed = self.result_store.mark_selection_committed(pending.battle_id)
        self.assertEqual("awaiting_terminal", committed.phase)

    def test_neutral_win_loss_and_tie_finalize(self):
        for index, outcome in enumerate(
            (TEAM_OUTCOME_A_WIN, TEAM_OUTCOME_A_LOSS, TEAM_OUTCOME_TIE), start=1
        ):
            with self.subTest(outcome=outcome):
                if index == 1:
                    self.initialize()
                record = self.complete(
                    outcome,
                    reservation=format(index, "032x"),
                    room="battle-gen9tugs-{}".format(index),
                )
                self.assertEqual(outcome, record.outcome)
                self.assertEqual(index, record.sequence)

    def test_no_result_is_operator_only_and_advances_hash_prefix(self):
        self.initialize()
        pending = self.pending()
        case = self.result_store.recovery_case()
        record = self.result_store.resolve_pending(case, TEAM_OUTCOME_NO_RESULT)
        self.assertEqual("operator_recovery", record.resolution_source)
        self.assertEqual(1, record.sequence)
        self.assertNotEqual("0" * 64, record.record_hash)
        self.assert_code(
            "team_result_terminal_outcome_invalid",
            lambda: self.result_store.finalize_terminal(
                pending.battle_id, TEAM_OUTCOME_NO_RESULT
            ),
        )

    def test_battle_id_is_deterministic_and_has_no_token_input(self):
        arguments = dict(
            reservation_id="a" * 32,
            room_id="battle-gen9tugs-1",
            player_account_id="player",
            bot_account_id="bot",
            player_team_id=PLAYER_TEAM_A,
            bot_team_id=BOT_A,
            format_id="gen9tugs",
            registry_fingerprint=FINGERPRINT,
        )
        first = derive_blind_team_battle_id(**arguments)
        second = derive_blind_team_battle_id(**arguments)
        self.assertEqual(first, second)
        self.assertNotIn("challenge", json.dumps(arguments))

    def test_account_ids_are_audit_metadata_not_rating_identity(self):
        self.initialize()
        record = self.complete(player_account="oldname")
        result = record.to_rating_result()
        self.assertEqual(PLAYER_TEAM_A, result.team_a_id)
        self.assertEqual(BOT_A, result.team_b_id)
        self.assertFalse(hasattr(result, "player_account_id"))

    def test_duplicate_terminal_is_idempotent_and_conflict_fails(self):
        self.initialize()
        record = self.complete()
        same = self.result_store.finalize_terminal(record.battle_id, TEAM_OUTCOME_A_WIN)
        self.assertEqual(record, same)
        self.assert_code(
            "team_result_outcome_conflict",
            lambda: self.result_store.finalize_terminal(
                record.battle_id, TEAM_OUTCOME_A_LOSS
            ),
        )

    def test_pending_blocks_ready_startup(self):
        self.initialize()
        self.pending()
        self.assert_code(
            "team_result_recovery_required", self.result_store.require_ready
        )

    def test_hash_chain_tampering_fails_closed(self):
        self.initialize()
        self.complete()
        document = json.loads(self.result_path.read_text(encoding="utf-8"))
        document["completed_results"][0]["outcome"] = TEAM_OUTCOME_A_LOSS
        self.result_path.write_text(json.dumps(document), encoding="utf-8")
        self.assert_code("team_result_record_hash_invalid", self.result_store.load)


class TeamPublicRegistryPersistenceTests(TeamPersistenceFixture):
    def test_randomized_aliases_are_unique_and_not_input_order_numbers(self):
        public, _, _ = self.initialize()
        self.assertEqual("Bot team 01", public.identity(BOT_B).display_name)
        self.assertEqual("Bot team 02", public.identity(BOT_A).display_name)
        self.assertEqual(2, len({item.display_name for item in public.identities}))
        self.assertNotIn(BOT_A, repr(public))

    def test_aliases_are_stable_across_reload(self):
        public, _, _ = self.initialize()
        self.assertEqual(public, self.public_store.load())

    def test_sync_preserves_aliases_and_issues_monotonic_new_alias(self):
        public, _, _ = self.initialize()
        before = public.identity(BOT_A).display_name
        expanded = self.public_store.sync_bot_identities(
            (BOT_A, "BL-003-v1"), random_source=IdentityRandom()
        )
        self.assertEqual(before, expanded.identity(BOT_A).display_name)
        self.assertEqual("Bot team 03", expanded.identity("BL-003-v1").display_name)
        self.assertIsNotNone(expanded.identity(BOT_B))

    def test_retired_alias_is_never_recycled(self):
        self.initialize()
        self.public_store.sync_bot_identities((BOT_A,), random_source=IdentityRandom())
        expanded = self.public_store.sync_bot_identities(
            (BOT_A, "BL-004-v1"), random_source=IdentityRandom()
        )
        self.assertEqual("Bot team 03", expanded.identity("BL-004-v1").display_name)

    def test_player_register_idempotent_and_rename_atomic(self):
        self.initialize()
        first = self.register_player()
        same = self.register_player()
        renamed = self.register_player(name="Arc-H Offense")
        self.assertEqual(first, same)
        self.assertEqual("Arc-H Offense", renamed.identity(PLAYER_TEAM_A).display_name)

    def test_duplicate_name_and_bot_namespace_fail_closed(self):
        self.initialize()
        self.register_player()
        self.assert_code(
            "team_public_name_duplicate",
            lambda: self.register_player(PLAYER_TEAM_B, "arc-h ho"),
        )
        self.assert_code(
            "team_public_name_reserved",
            lambda: BlindTeamPublicIdentity(
                PLAYER_TEAM_B, "Bot team 99", PUBLIC_TEAM_KIND_PLAYER
            ),
        )
        self.assert_code(
            "team_public_registry_kind_conflict",
            lambda: self.register_player(BOT_A, "Synthetic Player Team"),
        )

    def test_private_mapping_never_enters_public_leaderboard_rows(self):
        public, _, ratings = self.initialize()
        rows = build_leaderboard(ratings, public.public_registry)
        serialized = json.dumps([dict(row.to_public_dict()) for row in rows])
        self.assertNotIn(BOT_A, serialized)
        self.assertNotIn("private_team_id", serialized)


class TeamRatingPersistenceTests(TeamPersistenceFixture):
    def test_win_persists_exact_1516_1484_and_counters(self):
        self.initialize()
        self.complete()
        sync = self.rating_store.sync(self.result_store.require_ready())
        player = sync.state.team(PLAYER_TEAM_A)
        bot = sync.state.team(BOT_A)
        self.assertEqual(
            (1516, 1, 1, 0),
            (player.rating, player.games_played, player.wins, player.losses),
        )
        self.assertEqual(
            (1484, 1, 0, 1), (bot.rating, bot.games_played, bot.wins, bot.losses)
        )

    def test_bot_rating_carries_into_second_player_team_battle(self):
        self.initialize()
        self.complete()
        self.complete(
            player_team_id=PLAYER_TEAM_B,
            reservation="b" * 32,
            room="battle-gen9tugs-2",
        )
        state = self.rating_store.sync(self.result_store.require_ready()).state
        self.assertNotEqual(1484, state.team(BOT_A).rating)
        self.assertEqual(2, state.team(BOT_A).games_played)
        self.assertEqual(1, state.team(PLAYER_TEAM_B).games_played)

    def test_player_team_rating_carries_across_different_bot(self):
        self.initialize()
        self.complete()
        self.complete(
            bot_team_id=BOT_B,
            reservation="b" * 32,
            room="battle-gen9tugs-2",
        )
        state = self.rating_store.sync(self.result_store.require_ready()).state
        self.assertEqual(2, state.team(PLAYER_TEAM_A).games_played)
        self.assertEqual(1, state.team(BOT_B).games_played)

    def test_account_rename_does_not_create_rating_identity(self):
        self.initialize()
        self.complete(player_account="oldname")
        self.complete(
            player_account="newname",
            reservation="b" * 32,
            room="battle-gen9tugs-2",
        )
        state = self.rating_store.sync(self.result_store.require_ready()).state
        self.assertEqual(2, state.team(PLAYER_TEAM_A).games_played)
        self.assertFalse(hasattr(state, "players"))

    def test_no_result_advances_prefix_without_rating_entries(self):
        self.initialize()
        self.pending()
        self.result_store.resolve_pending(
            self.result_store.recovery_case(), TEAM_OUTCOME_NO_RESULT
        )
        state = self.rating_store.sync(self.result_store.require_ready()).state
        self.assertEqual(
            (1, 0, 0), (state.processed_sequence, state.rated_results, state.team_count)
        )

    def test_behind_catches_up_once_and_exact_state_remains_unchanged(self):
        self.initialize()
        self.complete()
        first = self.rating_store.sync(self.result_store.require_ready())
        second = self.rating_store.sync(self.result_store.require_ready())
        self.assertEqual(1, first.applied_results)
        self.assertEqual(0, second.applied_results)
        self.assertEqual(first.state, second.state)

    def test_ahead_state_fails_closed(self):
        self.initialize()
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["processed_sequence"] = 1
        document["processed_record_hash"] = "e" * 64
        self.rating_path.write_text(json.dumps(document), encoding="utf-8")
        self.assert_code(
            "team_rating_state_ahead",
            lambda: self.rating_store.status(self.result_store.require_ready()),
        )

    def test_valid_but_divergent_state_fails_closed(self):
        self.initialize()
        self.complete()
        self.rating_store.sync(self.result_store.require_ready())
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["teams"][PLAYER_TEAM_A]["rating"] = 1515
        document["teams"][BOT_A]["rating"] = 1485
        self.rating_path.write_text(json.dumps(document), encoding="utf-8")
        self.assert_code(
            "team_rating_state_diverged",
            lambda: self.rating_store.status(self.result_store.require_ready()),
        )

    def test_corrupt_rating_state_fails_closed(self):
        self.initialize()
        self.rating_path.write_bytes(b"{not-json")
        self.assert_code(
            "team_rating_json_invalid",
            lambda: self.rating_store.status(self.result_store.require_ready()),
        )

    def test_rebuild_is_deterministic(self):
        self.initialize()
        self.complete()
        expected = self.rating_store.sync(self.result_store.require_ready()).state
        rebuilt = self.rating_store.rebuild(self.result_store.require_ready())
        self.assertEqual(expected, rebuilt)

    def test_unplayed_registered_teams_appear_at_baseline(self):
        public, _, ratings = self.initialize()
        rows = build_leaderboard(ratings, public.public_registry)
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row.rating == 1500 and row.games == 0 for row in rows))

    def test_rating_state_is_immutable_under_leaderboard_projection(self):
        public, _, ratings = self.initialize()
        before = replace(ratings)
        build_leaderboard(ratings, public.public_registry)
        self.assertEqual(before, ratings)


if __name__ == "__main__":
    unittest.main()
