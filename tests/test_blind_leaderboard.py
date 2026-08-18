from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
import unittest

from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.leaderboard import (
    MAX_PUBLIC_TEAM_NAME_LENGTH,
    PUBLIC_TEAM_KIND_BOT,
    PUBLIC_TEAM_KIND_PLAYER,
    BlindLeaderboardRow,
    BlindTeamPublicIdentity,
    BlindTeamPublicRegistry,
    build_leaderboard,
)
from fp.data.blind_pool.rating import STREAK_LOSS, STREAK_NONE, STREAK_WIN
from fp.data.blind_pool.team_rating import (
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_TIE,
    TEAM_RATING_STATE_SCHEMA_VERSION,
    BlindTeamBattleResult,
    BlindTeamRating,
    BlindTeamRatingState,
    derive_team_rating_state,
    empty_team_rating_state,
)


TEAM_A = "synthetic-team-alpha"
TEAM_B = "synthetic-team-bravo"
TEAM_C = "synthetic-team-charlie"
TEAM_D = "synthetic-team-delta"
PRIVATE_SENTINEL = "synthetic-private-team-sentinel"
ZERO_HASH = "0" * 64


def identity(team_id, name, kind=PUBLIC_TEAM_KIND_PLAYER):
    return BlindTeamPublicIdentity(team_id, name, kind)


def result(sequence, team_a=TEAM_A, team_b=TEAM_B, outcome=TEAM_OUTCOME_A_WIN):
    return BlindTeamBattleResult(
        sequence,
        ZERO_HASH if sequence == 1 else format(sequence + 999, "064x"),
        format(sequence + 1000, "064x"),
        team_a,
        team_b,
        outcome,
    )


class PublicTeamIdentityTests(unittest.TestCase):
    def assert_code(self, code, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        self.assertNotIn(
            PRIVATE_SENTINEL, str(caught.exception) + repr(caught.exception)
        )

    def test_public_and_private_identity_are_explicitly_separate(self):
        public = identity(PRIVATE_SENTINEL, "Arc-H HO")
        self.assertEqual("Arc-H HO", public.display_name)
        self.assertEqual(PRIVATE_SENTINEL, public.team_id)
        self.assertNotIn(PRIVATE_SENTINEL, repr(public))
        self.assertIn("Arc-H HO", repr(public))

    def test_safe_unicode_names_and_documented_length_are_supported(self):
        self.assertEqual(40, MAX_PUBLIC_TEAM_NAME_LENGTH)
        public = identity(TEAM_A, "Éclair Balance")
        self.assertEqual("Éclair Balance", public.display_name)

    def test_empty_untrimmed_noncanonical_control_and_markup_names_fail(self):
        invalid_names = (
            "",
            " Leading",
            "Trailing ",
            "Double  Space",
            "Line\nBreak",
            "Tab\tName",
            "Unsafe <Team>",
            "A&B",
            "e\u0301",
            "x" * (MAX_PUBLIC_TEAM_NAME_LENGTH + 1),
            "bad\ud800name",
        )
        for public_name in invalid_names:
            with self.subTest(public_name=ascii(public_name)):
                self.assert_code(
                    "team_public_name_invalid",
                    lambda public_name=public_name: identity(TEAM_A, public_name),
                )

    def test_private_id_and_kind_are_strict(self):
        self.assert_code(
            "team_public_id_invalid",
            lambda: identity("not a stable id", "Safe Name"),
        )
        self.assert_code(
            "team_public_kind_invalid",
            lambda: identity(TEAM_A, "Safe Name", "account"),
        )

    def test_registry_rejects_duplicate_private_ids_and_public_names(self):
        self.assert_code(
            "team_public_id_duplicate",
            lambda: BlindTeamPublicRegistry(
                (identity(TEAM_A, "First"), identity(TEAM_A, "Second"))
            ),
        )
        self.assert_code(
            "team_public_name_duplicate",
            lambda: BlindTeamPublicRegistry(
                (identity(TEAM_A, "Arc-H HO"), identity(TEAM_B, "arc-h ho"))
            ),
        )

    def test_registry_repr_exposes_no_mapping(self):
        registry = BlindTeamPublicRegistry(
            (identity(PRIVATE_SENTINEL, "Public Alias"),)
        )
        rendered = repr(registry)
        self.assertEqual("BlindTeamPublicRegistry(team_count=1)", rendered)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertNotIn("Public Alias", rendered)


class LeaderboardProjectionTests(unittest.TestCase):
    def test_rated_and_unrated_registered_teams_both_appear(self):
        state = derive_team_rating_state((result(1),))
        rows = build_leaderboard(
            state,
            (
                identity(TEAM_A, "Arc-H HO"),
                identity(TEAM_B, "Bot team 01", PUBLIC_TEAM_KIND_BOT),
                identity(TEAM_C, "Drapion Offense"),
            ),
        )
        self.assertEqual(3, len(rows))
        unrated = next(row for row in rows if row.name == "Drapion Offense")
        self.assertEqual(
            (1500, 0, 0, 0, 0, "0.0%"),
            (
                unrated.rating,
                unrated.games,
                unrated.wins,
                unrated.losses,
                unrated.ties,
                unrated.win_rate_display,
            ),
        )

    def test_rating_is_primary_sort_and_rank_is_final_position(self):
        state = derive_team_rating_state((result(1),))
        rows = build_leaderboard(
            state,
            (
                identity(TEAM_B, "Bot team 01", PUBLIC_TEAM_KIND_BOT),
                identity(TEAM_C, "Drapion Offense"),
                identity(TEAM_A, "Arc-H HO"),
            ),
        )
        self.assertEqual(
            [
                (1, "Arc-H HO", 1516),
                (2, "Drapion Offense", 1500),
                (3, "Bot team 01", 1484),
            ],
            [(row.rank, row.name, row.rating) for row in rows],
        )

    def test_games_wins_and_normalized_name_are_deterministic_tie_breakers(self):
        teams = (
            BlindTeamRating(TEAM_A, 1500, 2, 1, 1, 0, 1500, STREAK_LOSS, 1),
            BlindTeamRating(TEAM_B, 1500, 2, 0, 0, 2, 1500, STREAK_NONE, 0),
            BlindTeamRating(TEAM_C, 1500, 2, 1, 1, 0, 1500, STREAK_WIN, 1),
            BlindTeamRating(TEAM_D, 1500, 2, 0, 0, 2, 1500, STREAK_NONE, 0),
        )
        state = BlindTeamRatingState(
            TEAM_RATING_STATE_SCHEMA_VERSION,
            "elo-v1",
            1500,
            32,
            400,
            "half-away-from-zero",
            4,
            "f" * 64,
            4,
            teams,
        )
        rows = build_leaderboard(
            state,
            (
                identity(TEAM_D, "Zulu"),
                identity(TEAM_B, "Alpha"),
                identity(TEAM_C, "Charlie"),
                identity(TEAM_A, "Bravo"),
            ),
        )
        self.assertEqual(
            ["Bravo", "Charlie", "Alpha", "Zulu"],
            [row.name for row in rows],
        )

    def test_more_games_precedes_an_unrated_team_at_equal_rating(self):
        state = derive_team_rating_state((result(1, outcome=TEAM_OUTCOME_TIE),))
        rows = build_leaderboard(
            state,
            (
                identity(TEAM_C, "A Unrated"),
                identity(TEAM_B, "Z Rated"),
                identity(TEAM_A, "M Rated"),
            ),
        )
        self.assertEqual(
            ["M Rated", "Z Rated", "A Unrated"], [row.name for row in rows]
        )

    def test_public_rename_changes_only_the_projected_name(self):
        state = derive_team_rating_state((result(1),))
        before = build_leaderboard(
            state,
            (
                identity(TEAM_A, "Arc-H HO"),
                identity(TEAM_B, "Bot team 01", PUBLIC_TEAM_KIND_BOT),
            ),
        )
        after = build_leaderboard(
            state,
            (
                identity(TEAM_A, "Webs Sun"),
                identity(TEAM_B, "Bot team 01", PUBLIC_TEAM_KIND_BOT),
            ),
        )
        self.assertEqual(
            (before[0].rating, before[0].games), (after[0].rating, after[0].games)
        )
        self.assertEqual(("Arc-H HO", "Webs Sun"), (before[0].name, after[0].name))
        self.assertEqual(1516, state.team(TEAM_A).rating)

    def test_player_and_bot_aliases_have_identical_row_semantics(self):
        rows = build_leaderboard(
            empty_team_rating_state(),
            (
                identity(TEAM_A, "Player Team", PUBLIC_TEAM_KIND_PLAYER),
                identity(TEAM_B, "Bot team 31", PUBLIC_TEAM_KIND_BOT),
            ),
        )
        self.assertEqual(
            {PUBLIC_TEAM_KIND_PLAYER, PUBLIC_TEAM_KIND_BOT}, {row.kind for row in rows}
        )
        for row in rows:
            self.assertEqual(
                (1500, 0, "0.0%"), (row.rating, row.games, row.win_rate_display)
            )

    def test_rows_and_public_serialization_contain_all_public_fields_and_no_id(self):
        rows = build_leaderboard(
            empty_team_rating_state(),
            (identity(PRIVATE_SENTINEL, "Bot team 31", PUBLIC_TEAM_KIND_BOT),),
        )
        row = rows[0]
        row_fields = {entry.name for entry in fields(BlindLeaderboardRow)}
        self.assertEqual(
            {
                "rank",
                "name",
                "kind",
                "rating",
                "games",
                "wins",
                "losses",
                "ties",
                "win_rate",
                "peak_rating",
                "streak",
            },
            row_fields,
        )
        public = row.to_public_dict()
        self.assertEqual("0.0%", public["win_rate"])
        self.assertEqual("Bot team 31", public["name"])
        self.assertNotIn("team_id", public)
        self.assertNotIn(PRIVATE_SENTINEL, repr(row) + repr(public))

    def test_projection_does_not_mutate_rating_state(self):
        state = derive_team_rating_state((result(1),))
        before = state
        build_leaderboard(
            state,
            (
                identity(TEAM_A, "Arc-H HO"),
                identity(TEAM_B, "Bot team 01", PUBLIC_TEAM_KIND_BOT),
            ),
        )
        self.assertIs(before, state)
        self.assertEqual(before, state)

    def test_missing_public_identity_for_rated_team_fails_closed(self):
        state = derive_team_rating_state((result(1),))
        with self.assertRaises(BlindPoolValidationError) as caught:
            build_leaderboard(state, (identity(TEAM_A, "Arc-H HO"),))
        self.assertEqual("leaderboard_identity_missing", caught.exception.code)
        rendered = str(caught.exception) + repr(caught.exception)
        self.assertNotIn(TEAM_B, rendered)

    def test_precise_win_rate_and_one_decimal_display_are_public_safe(self):
        row = BlindLeaderboardRow(
            1,
            "Drapion Offense",
            PUBLIC_TEAM_KIND_PLAYER,
            1864,
            7,
            4,
            2,
            1,
            Decimal(400) / Decimal(7),
            1900,
            "W2",
        )
        self.assertGreater(row.win_rate, Decimal("57.1"))
        self.assertEqual("57.1%", row.win_rate_display)


if __name__ == "__main__":
    unittest.main()
