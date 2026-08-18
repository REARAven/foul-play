from __future__ import annotations

from dataclasses import fields
from decimal import Decimal, localcontext
import unittest

from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.rating import (
    INITIAL_RATING,
    K_FACTOR,
    RATING_ALGORITHM_ID,
    RATING_SCALE,
    ROUNDING_POLICY,
    STREAK_LOSS,
    STREAK_NONE,
    STREAK_WIN,
    round_half_away_from_zero,
)
from fp.data.blind_pool.team_rating import (
    TEAM_OUTCOME_A_LOSS,
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_NO_RESULT,
    TEAM_OUTCOME_TIE,
    TEAM_RATING_STATE_SCHEMA_VERSION,
    ZERO_TEAM_HISTORY_HASH,
    BlindTeamBattleResult,
    BlindTeamRating,
    BlindTeamRatingState,
    apply_team_battle_result,
    baseline_team_rating,
    calculate_team_elo_delta,
    derive_team_rating_state,
    empty_team_rating_state,
    expected_team_score,
    validate_team_rating_state,
)


TEAM_A = "synthetic-team-alpha"
TEAM_B = "synthetic-team-bravo"
TEAM_C = "synthetic-team-charlie"
TEAM_D = "synthetic-team-delta"
PRIVATE_SENTINEL = "synthetic-private-team-sentinel"


def history_hash(sequence: int) -> str:
    return format(sequence + 1000, "064x")


def result(
    sequence: int,
    team_a: str = TEAM_A,
    team_b: str = TEAM_B,
    outcome: str = TEAM_OUTCOME_A_WIN,
    *,
    previous_hash: str | None = None,
    record_hash: str | None = None,
) -> BlindTeamBattleResult:
    if previous_hash is None:
        previous_hash = (
            ZERO_TEAM_HISTORY_HASH if sequence == 1 else history_hash(sequence - 1)
        )
    return BlindTeamBattleResult(
        sequence,
        previous_hash,
        record_hash or history_hash(sequence),
        team_a,
        team_b,
        outcome,
    )


class TeamEloContractTests(unittest.TestCase):
    def test_algorithm_contract_reuses_proven_elo_v1_constants(self):
        self.assertEqual(1, TEAM_RATING_STATE_SCHEMA_VERSION)
        self.assertEqual("elo-v1", RATING_ALGORITHM_ID)
        self.assertEqual(1500, INITIAL_RATING)
        self.assertEqual(32, K_FACTOR)
        self.assertEqual(400, RATING_SCALE)
        self.assertEqual("half-away-from-zero", ROUNDING_POLICY)

    def test_equal_rating_win_loss_and_tie_vectors_are_pinned(self):
        self.assertEqual(16, calculate_team_elo_delta(1500, 1500, TEAM_OUTCOME_A_WIN))
        self.assertEqual(
            -16,
            calculate_team_elo_delta(1500, 1500, TEAM_OUTCOME_A_LOSS),
        )
        self.assertEqual(0, calculate_team_elo_delta(1500, 1500, TEAM_OUTCOME_TIE))

    def test_unequal_expected_upset_and_expected_win_vectors(self):
        self.assertLess(expected_team_score(1400, 1600), Decimal("0.5"))
        upset = calculate_team_elo_delta(1400, 1600, TEAM_OUTCOME_A_WIN)
        expected = calculate_team_elo_delta(1600, 1400, TEAM_OUTCOME_A_WIN)
        self.assertEqual((24, 8), (upset, expected))
        self.assertGreater(upset, expected)

    def test_tie_transfer_and_each_applied_result_are_exactly_zero_sum(self):
        self.assertEqual(8, calculate_team_elo_delta(1400, 1600, TEAM_OUTCOME_TIE))
        state, update = apply_team_battle_result(
            empty_team_rating_state(),
            result(1),
        )
        self.assertEqual(0, update.team_a_delta + update.team_b_delta)
        self.assertEqual(3000, sum(team.rating for team in state.teams))

    def test_half_away_from_zero_policy_is_retained(self):
        self.assertEqual(3, round_half_away_from_zero(Decimal("2.5")))
        self.assertEqual(-3, round_half_away_from_zero(Decimal("-2.5")))

    def test_no_result_is_not_rateable(self):
        with self.assertRaises(BlindPoolValidationError) as caught:
            calculate_team_elo_delta(1500, 1500, TEAM_OUTCOME_NO_RESULT)
        self.assertEqual("team_rating_outcome_invalid", caught.exception.code)


class TeamRatingProjectionTests(unittest.TestCase):
    def test_empty_input_is_a_strict_empty_state(self):
        state = derive_team_rating_state(())
        self.assertEqual(empty_team_rating_state(), state)
        self.assertEqual(
            (0, 0, 0), (state.processed_sequence, state.rated_results, state.team_count)
        )

    def test_win_updates_both_team_counters_rating_and_peak(self):
        state, update = apply_team_battle_result(empty_team_rating_state(), result(1))
        team_a = state.team(TEAM_A)
        team_b = state.team(TEAM_B)
        self.assertIs(type(team_a), BlindTeamRating)
        self.assertIs(type(team_b), BlindTeamRating)
        self.assertEqual(
            (1516, 1, 1, 0, 0, 1516),
            (
                team_a.rating,
                team_a.games_played,
                team_a.wins,
                team_a.losses,
                team_a.ties,
                team_a.peak_rating,
            ),
        )
        self.assertEqual(
            (1484, 1, 0, 1, 0, 1500),
            (
                team_b.rating,
                team_b.games_played,
                team_b.wins,
                team_b.losses,
                team_b.ties,
                team_b.peak_rating,
            ),
        )
        self.assertEqual((16, -16), (update.team_a_delta, update.team_b_delta))

    def test_loss_and_tie_update_both_sides_symmetrically(self):
        loss_state = derive_team_rating_state((result(1, outcome=TEAM_OUTCOME_A_LOSS),))
        self.assertEqual(
            (1, 0), (loss_state.team(TEAM_A).losses, loss_state.team(TEAM_A).wins)
        )
        self.assertEqual(
            (1, 0), (loss_state.team(TEAM_B).wins, loss_state.team(TEAM_B).losses)
        )
        tie_state = derive_team_rating_state((result(1, outcome=TEAM_OUTCOME_TIE),))
        self.assertEqual(
            (1, 1), (tie_state.team(TEAM_A).ties, tie_state.team(TEAM_B).ties)
        )
        self.assertEqual(
            (1500, 1500), (tie_state.team(TEAM_A).rating, tie_state.team(TEAM_B).rating)
        )

    def test_counters_streaks_and_peak_follow_order_for_both_teams(self):
        results = (
            result(1, TEAM_A, TEAM_B, TEAM_OUTCOME_A_WIN),
            result(2, TEAM_A, TEAM_C, TEAM_OUTCOME_A_WIN),
            result(3, TEAM_A, TEAM_B, TEAM_OUTCOME_A_LOSS),
            result(4, TEAM_A, TEAM_C, TEAM_OUTCOME_A_LOSS),
        )
        before_tie = derive_team_rating_state(results)
        team_a = before_tie.team(TEAM_A)
        self.assertEqual(
            (4, 2, 2, 0), (team_a.games_played, team_a.wins, team_a.losses, team_a.ties)
        )
        self.assertEqual((STREAK_LOSS, 2), (team_a.streak_kind, team_a.streak_length))
        self.assertGreater(team_a.peak_rating, team_a.rating)
        after_tie = derive_team_rating_state(
            (*results, result(5, TEAM_A, TEAM_B, TEAM_OUTCOME_TIE))
        )
        self.assertEqual(
            (STREAK_NONE, 0),
            (
                after_tie.team(TEAM_A).streak_kind,
                after_tie.team(TEAM_A).streak_length,
            ),
        )
        self.assertEqual(
            (STREAK_NONE, 0),
            (
                after_tie.team(TEAM_B).streak_kind,
                after_tie.team(TEAM_B).streak_length,
            ),
        )
        self.assertEqual(team_a.peak_rating, after_tie.team(TEAM_A).peak_rating)

    def test_win_rate_is_derived_from_integer_counters(self):
        with localcontext() as context:
            context.prec = 50
            one_third_rate = Decimal(100) / Decimal(3)
        cases = (
            (
                BlindTeamRating(TEAM_A, 1500, 10, 7, 2, 1, 1500, STREAK_NONE, 0),
                Decimal("70"),
                "70.0%",
            ),
            (
                BlindTeamRating(TEAM_A, 1500, 3, 3, 0, 0, 1500, STREAK_WIN, 3),
                Decimal("100"),
                "100.0%",
            ),
            (baseline_team_rating(TEAM_A), Decimal("0"), "0.0%"),
            (
                BlindTeamRating(TEAM_A, 1500, 3, 1, 0, 2, 1500, STREAK_NONE, 0),
                one_third_rate,
                "33.3%",
            ),
        )
        for team, expected_rate, expected_display in cases:
            with self.subTest(record=(team.wins, team.losses, team.ties)):
                self.assertEqual(expected_rate, team.win_rate_percent)
                self.assertEqual(expected_display, team.win_rate_display)
        self.assertNotIn("win_rate", {entry.name for entry in fields(BlindTeamRating)})

    def test_no_result_advances_only_the_processed_prefix(self):
        no_result = result(1, outcome=TEAM_OUTCOME_NO_RESULT)
        state, update = apply_team_battle_result(empty_team_rating_state(), no_result)
        self.assertIsNone(update)
        self.assertEqual(
            (1, no_result.record_hash, 0, ()),
            (
                state.processed_sequence,
                state.processed_record_hash,
                state.rated_results,
                state.teams,
            ),
        )

    def test_replay_is_deterministic_and_full_rebuild_is_stable(self):
        results = (
            result(1),
            result(2, TEAM_C, TEAM_B, TEAM_OUTCOME_A_WIN),
            result(3, TEAM_A, TEAM_C, TEAM_OUTCOME_TIE),
        )
        first = derive_team_rating_state(results)
        second = derive_team_rating_state(tuple(results))
        self.assertEqual(first, second)
        self.assertEqual(history_hash(3), first.processed_record_hash)

    def test_bot_rating_carries_across_player_teams_in_one_peer_pool(self):
        state = derive_team_rating_state(
            (
                result(1, TEAM_A, TEAM_B),
                result(2, TEAM_C, TEAM_B),
            )
        )
        self.assertEqual(1516, state.team(TEAM_A).rating)
        self.assertEqual(1515, state.team(TEAM_C).rating)
        self.assertEqual(1469, state.team(TEAM_B).rating)
        self.assertTrue(all(type(team) is BlindTeamRating for team in state.teams))

    def test_player_team_carries_across_bot_teams_and_teams_are_independent(self):
        state = derive_team_rating_state(
            (
                result(1, TEAM_A, TEAM_B),
                result(2, TEAM_A, TEAM_D),
            )
        )
        self.assertEqual(1531, state.team(TEAM_A).rating)
        self.assertEqual(1484, state.team(TEAM_B).rating)
        self.assertEqual(1485, state.team(TEAM_D).rating)
        self.assertIsNone(state.team(TEAM_C))

    def test_account_identity_is_absent_from_the_team_model(self):
        for model in (BlindTeamRating, BlindTeamRatingState, BlindTeamBattleResult):
            names = {entry.name for entry in fields(model)}
            self.assertNotIn("player_id", names)
            self.assertNotIn("account_id", names)

    def test_private_team_id_is_redacted_from_all_internal_reprs(self):
        private_result = result(1, PRIVATE_SENTINEL, TEAM_B)
        state, _update = apply_team_battle_result(
            empty_team_rating_state(), private_result
        )
        rendered = (
            repr(private_result) + repr(state) + repr(state.team(PRIVATE_SENTINEL))
        )
        self.assertNotIn(PRIVATE_SENTINEL, rendered)


class TeamResultValidationTests(unittest.TestCase):
    def assert_code(self, code, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        self.assertNotIn(
            PRIVATE_SENTINEL, str(caught.exception) + repr(caught.exception)
        )

    def test_two_distinct_valid_stable_team_ids_are_required(self):
        self.assert_code(
            "team_result_same_team",
            lambda: result(1, PRIVATE_SENTINEL, PRIVATE_SENTINEL),
        )
        for invalid in ("", "has spaces", "x/unsafe", "x" * 129):
            with self.subTest(invalid=invalid):
                self.assert_code(
                    "team_result_team_id_invalid",
                    lambda invalid=invalid: result(1, invalid, TEAM_B),
                )

    def test_duplicate_and_out_of_order_results_are_rejected(self):
        first = result(1)
        state, _update = apply_team_battle_result(empty_team_rating_state(), first)
        self.assert_code(
            "team_result_sequence_invalid",
            lambda: apply_team_battle_result(state, first),
        )
        self.assert_code(
            "team_result_sequence_invalid",
            lambda: apply_team_battle_result(
                empty_team_rating_state(),
                result(2),
            ),
        )

    def test_previous_hash_mismatch_is_rejected(self):
        state, _update = apply_team_battle_result(empty_team_rating_state(), result(1))
        mismatched = result(2, previous_hash="f" * 64)
        self.assert_code(
            "team_result_sequence_invalid",
            lambda: apply_team_battle_result(state, mismatched),
        )

    def test_invalid_hashes_outcomes_and_history_types_are_rejected(self):
        self.assert_code(
            "team_result_hash_invalid",
            lambda: result(1, previous_hash="f" * 64),
        )
        self.assert_code(
            "team_result_hash_invalid",
            lambda: result(1, record_hash=ZERO_TEAM_HISTORY_HASH),
        )
        self.assert_code(
            "team_result_outcome_invalid",
            lambda: result(1, outcome="player_win"),
        )
        self.assert_code(
            "team_rating_history_invalid",
            lambda: derive_team_rating_state("not-a-history"),
        )

    def test_state_validation_rejects_nonconservation_and_duplicates(self):
        state = derive_team_rating_state((result(1),))
        broken = BlindTeamRatingState(
            state.schema_version,
            state.algorithm_id,
            state.initial_rating,
            state.k_factor,
            state.scale,
            state.rounding_policy,
            state.processed_sequence,
            state.processed_record_hash,
            state.rated_results,
            (state.teams[0], state.teams[0]),
        )
        self.assert_code(
            "team_rating_team_duplicate",
            lambda: validate_team_rating_state(broken),
        )
        altered = BlindTeamRating(
            state.teams[0].team_id,
            state.teams[0].rating + 1,
            state.teams[0].games_played,
            state.teams[0].wins,
            state.teams[0].losses,
            state.teams[0].ties,
            state.teams[0].peak_rating + 1,
            state.teams[0].streak_kind,
            state.teams[0].streak_length,
        )
        nonconserved = BlindTeamRatingState(
            state.schema_version,
            state.algorithm_id,
            state.initial_rating,
            state.k_factor,
            state.scale,
            state.rounding_policy,
            state.processed_sequence,
            state.processed_record_hash,
            state.rated_results,
            (altered, state.teams[1]),
        )
        self.assert_code(
            "team_rating_counter_inconsistent",
            lambda: validate_team_rating_state(nonconserved),
        )


if __name__ == "__main__":
    unittest.main()
