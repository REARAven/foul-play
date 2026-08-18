from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.locking import BlindPoolStateLock
from fp.data.blind_pool.rating import (
    INITIAL_RATING,
    K_FACTOR,
    RATING_ALGORITHM_ID,
    RATING_SCALE,
    RATING_STATE_SCHEMA_VERSION,
    ROUNDING_POLICY,
    STREAK_LOSS,
    STREAK_NONE,
    STREAK_WIN,
    baseline_player_rating,
    calculate_elo_delta,
    derive_rating_state,
    empty_rating_state,
    expected_player_score,
    round_half_away_from_zero,
)
from fp.data.blind_pool.rating_state import (
    RATING_STATUS_BEHIND,
    RATING_STATUS_SYNCED,
    BlindRatingStateStore,
    validate_rating_state_config,
    validate_rating_state_document,
)
from fp.data.blind_pool.result_ledger import (
    OUTCOME_NO_RESULT,
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
)
from tests.test_blind_result_ledger import ResultLedgerFixture


PRIVATE_SENTINEL = "PHASE6B-SYNTHETIC-PRIVATE-SENTINEL"


class RatingFixture(ResultLedgerFixture):
    def setUp(self):
        super().setUp()
        self.initialize()
        self.rating_path = self.ladder_root / "ratings.json"
        self.rating_config = validate_rating_state_config(
            self.rating_path,
            private_root=self.private_root,
            registry_path=self.registry_path,
            selection_state_path=self.state_path,
            result_ledger_path=self.ledger_path,
        )
        self.rating_store = BlindRatingStateStore(
            self.rating_config,
            lock_timeout_seconds=0.1,
        )
        self.record_number = 0

    def complete(
        self,
        outcome=OUTCOME_PLAYER_WIN,
        *,
        player="syntheticplayer",
        team_id=None,
        operator=False,
    ):
        self.record_number += 1
        number = self.record_number
        pending = self.create_awaiting(
            player_id=player,
            team_id=team_id or "BL-{:03d}-v1".format(number),
            reservation_id=format(number, "032x"),
            room_id="battle-gen9tugs-{}".format(500 + number),
        )
        self.advance_clock()
        if operator or outcome == OUTCOME_NO_RESULT:
            return self.store.resolve_pending(self.store.recovery_case(), outcome)
        return self.store.finalize_terminal(pending.battle_id, outcome)

    def assert_rating_code(self, code, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        rendered = str(caught.exception) + repr(caught.exception)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertNotIn(str(self.root), rendered)


class EloFormulaTests(unittest.TestCase):
    def test_algorithm_contract_is_explicit_and_versioned(self):
        self.assertEqual(1, RATING_STATE_SCHEMA_VERSION)
        self.assertEqual("elo-v1", RATING_ALGORITHM_ID)
        self.assertEqual(1500, INITIAL_RATING)
        self.assertEqual(32, K_FACTOR)
        self.assertEqual(400, RATING_SCALE)
        self.assertEqual("half-away-from-zero", ROUNDING_POLICY)

    def test_equal_rating_vectors_are_pinned(self):
        self.assertEqual(16, calculate_elo_delta(1500, 1500, OUTCOME_PLAYER_WIN))
        self.assertEqual(-16, calculate_elo_delta(1500, 1500, OUTCOME_PLAYER_LOSS))
        self.assertEqual(0, calculate_elo_delta(1500, 1500, OUTCOME_TIE))

    def test_unequal_expected_upset_and_expected_win_vectors(self):
        self.assertLess(expected_player_score(1400, 1600), Decimal("0.5"))
        upset = calculate_elo_delta(1400, 1600, OUTCOME_PLAYER_WIN)
        expected = calculate_elo_delta(1600, 1400, OUTCOME_PLAYER_WIN)
        self.assertEqual(24, upset)
        self.assertEqual(8, expected)
        self.assertGreater(upset, expected)

    def test_unequal_ties_transfer_rating_in_both_directions(self):
        low = calculate_elo_delta(1400, 1600, OUTCOME_TIE)
        high = calculate_elo_delta(1600, 1400, OUTCOME_TIE)
        self.assertEqual(8, low)
        self.assertEqual(-8, high)
        self.assertEqual(0, low + high)

    def test_half_away_from_zero_is_explicit_at_boundaries(self):
        vectors = (
            ("2.5", 3),
            ("-2.5", -3),
            ("2.499999999", 2),
            ("-2.499999999", -2),
        )
        for raw, expected in vectors:
            with self.subTest(raw=raw):
                self.assertEqual(
                    expected,
                    round_half_away_from_zero(Decimal(raw)),
                )

    def test_pinned_rating_vectors_are_deterministic_integers(self):
        vectors = (
            (1200, 1800, OUTCOME_PLAYER_WIN, 31),
            (1800, 1200, OUTCOME_PLAYER_WIN, 1),
            (1200, 1800, OUTCOME_PLAYER_LOSS, -1),
            (1800, 1200, OUTCOME_PLAYER_LOSS, -31),
            (1473, 1638, OUTCOME_TIE, 7),
        )
        for player, opponent, outcome, expected in vectors:
            with self.subTest(vector=(player, opponent, outcome)):
                delta = calculate_elo_delta(player, opponent, outcome)
                self.assertIs(type(delta), int)
                self.assertEqual(expected, delta)


class RatingProjectionTests(RatingFixture):
    def test_empty_ledger_derives_strict_empty_projection(self):
        state = derive_rating_state(())
        self.assertEqual(empty_rating_state(), state)
        self.assertEqual(0, state.processed_results)
        self.assertEqual(0, state.rated_results)

    def test_one_result_updates_player_and_hidden_opponent_zero_sum(self):
        record = self.complete()
        state = derive_rating_state((record,))
        player = state.player("syntheticplayer")
        self.assertIsNotNone(player)
        self.assertEqual(1516, player.rating)
        opponent = state.opponents[0]
        self.assertEqual(1484, opponent.rating)
        self.assertEqual(3000, player.rating + opponent.rating)
        self.assertEqual(record.record_hash, state.processed_record_hash)

    def test_record_counters_streaks_and_peak_follow_order(self):
        self.complete(OUTCOME_PLAYER_WIN)
        self.complete(OUTCOME_PLAYER_WIN)
        state = derive_rating_state(self.store.load().completed_results)
        player = state.player("syntheticplayer")
        self.assertEqual(STREAK_WIN, player.streak_kind)
        self.assertEqual(2, player.streak_length)
        peak = player.peak_rating
        self.complete(OUTCOME_PLAYER_LOSS)
        self.complete(OUTCOME_PLAYER_LOSS)
        state = derive_rating_state(self.store.load().completed_results)
        player = state.player("syntheticplayer")
        self.assertEqual((2, 2, 0), (player.wins, player.losses, player.ties))
        self.assertEqual(player.games_played, player.wins + player.losses + player.ties)
        self.assertEqual(STREAK_LOSS, player.streak_kind)
        self.assertEqual(2, player.streak_length)
        self.assertEqual(peak, player.peak_rating)
        self.complete(OUTCOME_TIE)
        player = derive_rating_state(self.store.load().completed_results).player(
            "syntheticplayer"
        )
        self.assertEqual(STREAK_NONE, player.streak_kind)
        self.assertEqual(0, player.streak_length)

    def test_no_result_advances_prefix_without_rating(self):
        record = self.complete(OUTCOME_NO_RESULT)
        state = derive_rating_state((record,))
        self.assertEqual(1, state.processed_sequence)
        self.assertEqual(record.record_hash, state.processed_record_hash)
        self.assertEqual(0, state.rated_results)
        self.assertEqual((), state.players)
        self.assertEqual((), state.opponents)

    def test_operator_outcomes_are_rated_normally(self):
        for outcome in (OUTCOME_PLAYER_WIN, OUTCOME_PLAYER_LOSS, OUTCOME_TIE):
            self.complete(outcome, operator=True)
        state = derive_rating_state(self.store.load().completed_results)
        player = state.player("syntheticplayer")
        self.assertEqual(
            (3, 1, 1, 1),
            (
                player.games_played,
                player.wins,
                player.losses,
                player.ties,
            ),
        )

    def test_players_are_independent_but_hidden_opponent_is_shared(self):
        team = "BL-001-v1"
        self.complete(player="playera", team_id=team)
        first = derive_rating_state(self.store.load().completed_results)
        self.assertEqual(1484, first.opponents[0].rating)
        self.complete(player="playerb", team_id=team)
        state = derive_rating_state(self.store.load().completed_results)
        self.assertEqual(1515, state.player("playerb").rating)
        self.assertEqual(1469, state.opponents[0].rating)
        self.assertEqual(1516, state.player("playera").rating)

    def test_different_and_historical_team_ids_remain_independent(self):
        self.complete(team_id="BL-001-v1")
        self.complete(team_id="BL-999-v7")
        state = derive_rating_state(self.store.load().completed_results)
        self.assertEqual(2, len(state.opponents))
        self.assertEqual(
            {"BL-001-v1", "BL-999-v7"},
            {opponent.team_id for opponent in state.opponents},
        )

    def test_safe_representations_hide_opponent_identity_and_mappings(self):
        self.complete(team_id="BL-777-v9")
        state = derive_rating_state(self.store.load().completed_results)
        rendered = repr(state) + repr(state.opponents[0])
        self.assertNotIn("BL-777-v9", rendered)
        self.assertNotIn("1484", repr(state.opponents[0]))
        self.assertNotIn("opponents", repr(state))

    def test_baseline_player_is_unrated_and_private_identity_is_redacted(self):
        player = baseline_player_rating("newplayer")
        self.assertEqual(
            (1500, 0, 1500),
            (
                player.rating,
                player.games_played,
                player.peak_rating,
            ),
        )
        self.assertNotIn("newplayer", repr(player))


class RatingPersistenceTests(RatingFixture):
    def test_initialize_empty_and_nonempty_are_deterministic_and_idempotent(self):
        empty = self.rating_store.initialize(self.store.require_ready())
        self.assertEqual(empty_rating_state(), empty)
        self.assertEqual(
            empty, self.rating_store.initialize(self.store.require_ready())
        )
        self.rating_path.unlink()
        self.complete()
        expected = derive_rating_state(self.store.load().completed_results)
        self.assertEqual(
            expected, self.rating_store.initialize(self.store.require_ready())
        )
        self.assertEqual(expected, self.rating_store.load())
        raw = self.rating_path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)

    def test_conflicting_existing_initialization_is_rejected(self):
        self.rating_store.initialize(self.store.require_ready())
        self.complete()
        self.assert_rating_code(
            "rating_initialization_conflict",
            lambda: self.rating_store.initialize(self.store.require_ready()),
        )

    def test_synced_and_behind_status_and_exact_once_sync(self):
        self.rating_store.initialize(self.store.require_ready())
        status, _state = self.rating_store.status(self.store.require_ready())
        self.assertEqual(RATING_STATUS_SYNCED, status)
        self.complete()
        status, behind = self.rating_store.status(self.store.require_ready())
        self.assertEqual(RATING_STATUS_BEHIND, status)
        self.assertEqual(0, behind.processed_sequence)
        first = self.rating_store.sync(self.store.require_ready())
        self.assertEqual(1, len(first.updates))
        self.assertEqual(1516, first.last_update.rating_after)
        second = self.rating_store.sync(self.store.require_ready())
        self.assertEqual(RATING_STATUS_SYNCED, second.status_before)
        self.assertEqual((), second.updates)
        self.assertEqual(first.state, second.state)

    def test_multi_record_and_no_result_tail_catch_up_in_order(self):
        self.rating_store.initialize(self.store.require_ready())
        self.complete(OUTCOME_PLAYER_WIN)
        self.complete(OUTCOME_NO_RESULT)
        self.complete(OUTCOME_PLAYER_LOSS)
        synchronized = self.rating_store.sync(self.store.require_ready())
        self.assertEqual(3, synchronized.state.processed_sequence)
        self.assertEqual(2, synchronized.state.rated_results)
        self.assertEqual(2, len(synchronized.updates))
        self.assertEqual(3, synchronized.applied_results)
        self.assertEqual(
            derive_rating_state(self.store.load().completed_results),
            synchronized.state,
        )

    def test_prefix_hash_divergence_and_ahead_state_fail_closed(self):
        self.rating_store.initialize(self.store.require_ready())
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["processed_record_hash"] = "a" * 64
        self.rating_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        self.assert_rating_code(
            "rating_processed_prefix_invalid",
            lambda: self.rating_store.status(self.store.require_ready()),
        )
        document["processed_sequence"] = 1
        self.rating_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        self.assert_rating_code(
            "rating_state_ahead",
            lambda: self.rating_store.status(self.store.require_ready()),
        )

    def test_valid_but_tampered_rating_projection_is_diverged(self):
        self.complete()
        self.rating_store.initialize(self.store.require_ready())
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["players"]["syntheticplayer"]["rating"] += 1
        document["players"]["syntheticplayer"]["peak_rating"] += 1
        document["opponents"]["BL-001-v1"]["rating"] -= 1
        self.rating_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        self.assert_rating_code(
            "rating_state_diverged",
            lambda: self.rating_store.status(self.store.require_ready()),
        )

    def test_rebuild_repairs_derived_state_without_modifying_ledger(self):
        self.complete()
        expected = self.rating_store.initialize(self.store.require_ready())
        ledger_before = self.ledger_path.read_bytes()
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["players"]["syntheticplayer"]["rating"] += 1
        document["players"]["syntheticplayer"]["peak_rating"] += 1
        document["opponents"]["BL-001-v1"]["rating"] -= 1
        self.rating_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        self.assertEqual(
            expected, self.rating_store.rebuild(self.store.require_ready())
        )
        self.assertEqual(ledger_before, self.ledger_path.read_bytes())

    def test_strict_schema_missing_extra_duplicate_and_bad_encoding_are_rejected(self):
        self.rating_store.initialize(self.store.require_ready())
        valid = json.loads(self.rating_path.read_text(encoding="utf-8"))
        missing = dict(valid)
        missing.pop("rated_results")
        self.assert_rating_code(
            "rating_missing_required_field",
            lambda: validate_rating_state_document(missing),
        )
        extra = dict(valid, extra=True)
        self.assert_rating_code(
            "rating_unexpected_field",
            lambda: validate_rating_state_document(extra),
        )
        cases = (
            (b'{"schema_version":1,"schema_version":1}', "rating_duplicate_field"),
            (b"\xff", "rating_encoding_invalid"),
            (b"{", "rating_json_invalid"),
            (b"\xef\xbb\xbf{}", "rating_encoding_invalid"),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                self.rating_path.write_bytes(payload)
                self.assert_rating_code(code, self.rating_store.load)

    def test_invalid_algorithm_counters_streak_and_prefix_are_rejected(self):
        self.complete()
        self.rating_store.initialize(self.store.require_ready())
        valid = json.loads(self.rating_path.read_text(encoding="utf-8"))
        cases = (
            (
                lambda d: d["algorithm"].update(id="elo-v2"),
                "rating_algorithm_unsupported",
            ),
            (lambda d: d.update(schema_version=2), "rating_schema_unsupported"),
            (
                lambda d: d["players"]["syntheticplayer"].update(games_played=2),
                "rating_player_invalid",
            ),
            (
                lambda d: d["players"]["syntheticplayer"].update(
                    streak_kind="none", streak_length=1
                ),
                "rating_player_invalid",
            ),
            (
                lambda d: d["players"]["syntheticplayer"].update(
                    streak_kind="loss", streak_length=1
                ),
                "rating_player_invalid",
            ),
            (
                lambda d: d.update(processed_sequence=-1),
                "rating_processed_prefix_invalid",
            ),
        )
        for mutation, code in cases:
            document = json.loads(json.dumps(valid))
            mutation(document)
            with self.subTest(code=code):
                self.assert_rating_code(
                    code,
                    lambda document=document: validate_rating_state_document(document),
                )

    def test_external_path_deployment_repository_and_collisions_are_enforced(self):
        values = {
            "private_root": self.private_root,
            "registry_path": self.registry_path,
            "selection_state_path": self.state_path,
            "result_ledger_path": self.ledger_path,
        }
        cases = (
            (Path.cwd() / "ratings.json", "rating_path_not_external"),
            (self.private_root / "ratings.json", "rating_path_in_deployment"),
            (self.ledger_path, "rating_path_collision"),
            (
                self.rating_config.result_ledger_path.with_name(
                    self.ledger_path.name + ".lock"
                ),
                "rating_path_collision",
            ),
            (self.state_path, "rating_path_in_deployment"),
        )
        for path, code in cases:
            with self.subTest(code=code):
                self.assert_rating_code(
                    code,
                    lambda path=path: validate_rating_state_config(path, **values),
                )

    def test_symlink_and_hardlink_rating_files_are_rejected_when_supported(self):
        target = self.ladder_root / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        link = self.ladder_root / "link.json"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            link = None
        if link is not None:
            self.assert_rating_code(
                "rating_path_unsafe",
                lambda: validate_rating_state_config(
                    link,
                    private_root=self.private_root,
                    registry_path=self.registry_path,
                    selection_state_path=self.state_path,
                    result_ledger_path=self.ledger_path,
                ),
            )
        hardlink = self.ladder_root / "hardlink.json"
        try:
            hardlink.hardlink_to(target)
        except (NotImplementedError, OSError):
            return
        self.assert_rating_code(
            "rating_path_unsafe",
            lambda: validate_rating_state_config(
                hardlink,
                private_root=self.private_root,
                registry_path=self.registry_path,
                selection_state_path=self.state_path,
                result_ledger_path=self.ledger_path,
            ),
        )

    def test_dedicated_lock_serializes_concurrent_writers(self):
        self.rating_store.initialize(self.store.require_ready())
        observed = []

        def blocked_load():
            try:
                BlindRatingStateStore(
                    self.rating_config,
                    lock_timeout_seconds=0.02,
                ).load()
            except BlindPoolValidationError as error:
                observed.append(error.code)

        with BlindPoolStateLock(self.rating_config.lock_path, timeout_seconds=0.1):
            thread = threading.Thread(target=blocked_load)
            thread.start()
            thread.join()
        self.assertEqual(["state_lock_timeout"], observed)

    def test_concurrent_sync_writers_serialize_without_duplicate_rating(self):
        self.rating_store.initialize(self.store.require_ready())
        self.complete()
        ledger = self.store.require_ready()
        results = []
        errors = []

        def synchronize():
            try:
                results.append(
                    BlindRatingStateStore(
                        self.rating_config,
                        lock_timeout_seconds=1,
                    ).sync(ledger)
                )
            except BlindPoolValidationError as error:
                errors.append(error.code)

        threads = [threading.Thread(target=synchronize) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        self.assertEqual({0, 1}, {result.applied_results for result in results})
        final = self.rating_store.verify(ledger)
        self.assertEqual(
            (1, 1516),
            (
                final.rated_results,
                final.player("syntheticplayer").rating,
            ),
        )

    def test_atomic_sync_failure_preserves_prior_snapshot_and_cleans_temp(self):
        self.rating_store.initialize(self.store.require_ready())
        before = self.rating_path.read_bytes()
        self.complete()
        with mock.patch(
            "fp.data.blind_pool.rating_state.os.replace",
            side_effect=OSError("synthetic replace failure"),
        ):
            self.assert_rating_code(
                "rating_atomic_write_failed",
                lambda: self.rating_store.sync(self.store.require_ready()),
            )
        self.assertEqual(before, self.rating_path.read_bytes())
        self.assertEqual(
            [],
            list(self.rating_path.parent.glob(".ratings.json-*.tmp")),
        )
        synchronized = self.rating_store.sync(self.store.require_ready())
        self.assertEqual(1, synchronized.state.rated_results)

    def test_unresolved_result_pending_blocks_all_rating_projection(self):
        self.create_awaiting()
        self.assert_rating_code(
            "rating_ledger_not_ready",
            lambda: self.rating_store.initialize(self.store.load()),
        )

    def test_rating_update_contains_only_public_player_fields(self):
        self.rating_store.initialize(self.store.require_ready())
        self.complete(team_id="BL-555-v5")
        update = self.rating_store.sync(self.store.require_ready()).last_update
        rendered = repr(update)
        self.assertNotIn("BL-555-v5", rendered)
        self.assertNotIn("opponent", rendered.lower())
        self.assertEqual(1516, update.rating_after)
