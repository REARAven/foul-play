"""Dormant team-centric Elo-v1 projection for the Blind Ladder.

This pure projection deliberately has no production result-ledger or startup
wiring. Phase 6B.2 can supply authoritative identities for both teams without
changing the rating semantics established here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re
from typing import Any, NoReturn, Sequence

from .errors import BlindPoolValidationError
from .rating import (
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


TEAM_RATING_STATE_SCHEMA_VERSION = 1
ZERO_TEAM_HISTORY_HASH = "0" * 64

TEAM_OUTCOME_A_WIN = "team_a_win"
TEAM_OUTCOME_A_LOSS = "team_a_loss"
TEAM_OUTCOME_TIE = "tie"
TEAM_OUTCOME_NO_RESULT = "no_result"
TEAM_RESULT_OUTCOMES = frozenset(
    {
        TEAM_OUTCOME_A_WIN,
        TEAM_OUTCOME_A_LOSS,
        TEAM_OUTCOME_TIE,
        TEAM_OUTCOME_NO_RESULT,
    }
)

_RATED_TEAM_OUTCOMES = frozenset(
    {TEAM_OUTCOME_A_WIN, TEAM_OUTCOME_A_LOSS, TEAM_OUTCOME_TIE}
)
_TEAM_A_SCORES = {
    TEAM_OUTCOME_A_WIN: Decimal("1"),
    TEAM_OUTCOME_A_LOSS: Decimal("0"),
    TEAM_OUTCOME_TIE: Decimal("0.5"),
}
_PRIVATE_TEAM_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_STREAK_KINDS = frozenset({STREAK_NONE, STREAK_WIN, STREAK_LOSS})


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def is_valid_private_team_id(value: object) -> bool:
    """Return whether a value is a stable, private ladder-team identifier."""

    return isinstance(value, str) and _PRIVATE_TEAM_ID.fullmatch(value) is not None


def _validate_integer(value: Any, *, context: str) -> int:
    if type(value) is not int:
        _fail("team_rating_integer_invalid", "{} must be an integer".format(context))
    return value


def expected_team_score(team_a_rating: int, team_b_rating: int) -> Decimal:
    """Return Team A's deterministic high-precision Elo expected score."""

    team_a = _validate_integer(team_a_rating, context="Team A rating")
    team_b = _validate_integer(team_b_rating, context="Team B rating")
    with localcontext() as context:
        context.prec = 50
        exponent = Decimal(team_b - team_a) / Decimal(RATING_SCALE)
        return Decimal(1) / (Decimal(1) + (Decimal(10) ** exponent))


def calculate_team_elo_delta(
    team_a_rating: int,
    team_b_rating: int,
    outcome: str,
) -> int:
    """Return Team A's Elo-v1 delta; Team B always receives its exact inverse."""

    if not isinstance(outcome, str) or outcome not in _RATED_TEAM_OUTCOMES:
        _fail("team_rating_outcome_invalid", "Team outcome is not rateable")
    expected = expected_team_score(team_a_rating, team_b_rating)
    with localcontext() as context:
        context.prec = 50
        raw_delta = Decimal(K_FACTOR) * (_TEAM_A_SCORES[outcome] - expected)
        return round_half_away_from_zero(raw_delta)


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamRating:
    """One private-identity team rating shared by every ladder team kind."""

    team_id: str = field(repr=False)
    rating: int
    games_played: int
    wins: int
    losses: int
    ties: int
    peak_rating: int
    streak_kind: str
    streak_length: int

    @property
    def streak_label(self) -> str:
        if self.streak_kind == STREAK_WIN:
            return "W{}".format(self.streak_length)
        if self.streak_kind == STREAK_LOSS:
            return "L{}".format(self.streak_length)
        return "none"

    @property
    def win_rate_percent(self) -> Decimal:
        """Return wins/games as a deterministic percentage without storing it."""

        if self.games_played == 0:
            return Decimal("0")
        with localcontext() as context:
            context.prec = 50
            return Decimal(self.wins) * Decimal(100) / Decimal(self.games_played)

    @property
    def win_rate_display(self) -> str:
        rounded = self.win_rate_percent.quantize(
            Decimal("0.1"),
            rounding=ROUND_HALF_UP,
        )
        return "{:.1f}%".format(rounded)

    def __repr__(self) -> str:
        return (
            "BlindTeamRating(private=True, rating={!r}, games_played={!r}, "
            "wins={!r}, losses={!r}, ties={!r})"
        ).format(
            self.rating,
            self.games_played,
            self.wins,
            self.losses,
            self.ties,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindTeamRating serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamRatingState:
    """Immutable team-only projection of an ordered result prefix."""

    schema_version: int
    algorithm_id: str
    initial_rating: int
    k_factor: int
    scale: int
    rounding_policy: str
    processed_sequence: int
    processed_record_hash: str = field(repr=False)
    rated_results: int
    teams: tuple[BlindTeamRating, ...] = field(repr=False)

    @property
    def processed_results(self) -> int:
        return self.processed_sequence

    @property
    def team_count(self) -> int:
        return len(self.teams)

    def team(self, team_id: str) -> BlindTeamRating | None:
        if not is_valid_private_team_id(team_id):
            _fail("team_rating_team_id_invalid", "Team identity is invalid")
        return next((team for team in self.teams if team.team_id == team_id), None)

    def __repr__(self) -> str:
        return (
            "BlindTeamRatingState(processed_results={!r}, rated_results={!r}, "
            "team_count={!r})"
        ).format(
            self.processed_results,
            self.rated_results,
            self.team_count,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindTeamRatingState serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamBattleResult:
    """Neutral, two-team input for the dormant deterministic projection."""

    sequence: int
    previous_record_hash: str = field(repr=False)
    record_hash: str = field(repr=False)
    team_a_id: str = field(repr=False)
    team_b_id: str = field(repr=False)
    outcome: str

    def __post_init__(self) -> None:
        validate_team_battle_result(self)

    def __repr__(self) -> str:
        return (
            "BlindTeamBattleResult(sequence={!r}, outcome={!r}, private=True)".format(
                self.sequence,
                self.outcome,
            )
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindTeamBattleResult serialization is disabled")


@dataclass(frozen=True, slots=True)
class BlindTeamRatingUpdate:
    """Public-safe numeric update for both sides of one rated team result."""

    team_a_rating_before: int
    team_a_rating_after: int
    team_a_delta: int
    team_b_rating_before: int
    team_b_rating_after: int
    team_b_delta: int


def empty_team_rating_state() -> BlindTeamRatingState:
    return BlindTeamRatingState(
        TEAM_RATING_STATE_SCHEMA_VERSION,
        RATING_ALGORITHM_ID,
        INITIAL_RATING,
        K_FACTOR,
        RATING_SCALE,
        ROUNDING_POLICY,
        0,
        ZERO_TEAM_HISTORY_HASH,
        0,
        (),
    )


def baseline_team_rating(team_id: str) -> BlindTeamRating:
    if not is_valid_private_team_id(team_id):
        _fail("team_rating_team_id_invalid", "Team identity is invalid")
    return BlindTeamRating(
        team_id,
        INITIAL_RATING,
        0,
        0,
        0,
        0,
        INITIAL_RATING,
        STREAK_NONE,
        0,
    )


def _validate_team(team: BlindTeamRating) -> None:
    if not is_valid_private_team_id(team.team_id):
        _fail("team_rating_team_id_invalid", "Team identity is invalid")
    integers = (
        team.rating,
        team.games_played,
        team.wins,
        team.losses,
        team.ties,
        team.peak_rating,
        team.streak_length,
    )
    if any(type(value) is not int for value in integers):
        _fail("team_rating_entry_invalid", "Team rating state is invalid")
    if (
        min(team.games_played, team.wins, team.losses, team.ties, team.streak_length)
        < 0
        or team.games_played != team.wins + team.losses + team.ties
        or team.peak_rating < INITIAL_RATING
        or team.peak_rating < team.rating
        or team.streak_kind not in _STREAK_KINDS
        or (team.streak_kind == STREAK_NONE and team.streak_length != 0)
        or (team.streak_kind != STREAK_NONE and team.streak_length <= 0)
        or team.streak_length > team.games_played
        or (team.streak_kind == STREAK_WIN and team.streak_length > team.wins)
        or (team.streak_kind == STREAK_LOSS and team.streak_length > team.losses)
        or (
            team.games_played == 0
            and (
                team.rating != INITIAL_RATING
                or team.peak_rating != INITIAL_RATING
                or team.streak_kind != STREAK_NONE
            )
        )
    ):
        _fail("team_rating_entry_invalid", "Team rating state is inconsistent")


def validate_team_rating_state(state: BlindTeamRatingState) -> BlindTeamRatingState:
    """Validate one immutable, conserved team-centric Elo projection."""

    if not isinstance(state, BlindTeamRatingState):
        _fail("team_rating_state_invalid", "Team rating state is invalid")
    if (
        type(state.schema_version) is not int
        or state.schema_version != TEAM_RATING_STATE_SCHEMA_VERSION
        or state.algorithm_id != RATING_ALGORITHM_ID
        or type(state.initial_rating) is not int
        or state.initial_rating != INITIAL_RATING
        or type(state.k_factor) is not int
        or state.k_factor != K_FACTOR
        or type(state.scale) is not int
        or state.scale != RATING_SCALE
        or state.rounding_policy != ROUNDING_POLICY
    ):
        _fail("team_rating_algorithm_unsupported", "Rating algorithm is unsupported")
    if (
        type(state.processed_sequence) is not int
        or state.processed_sequence < 0
        or type(state.rated_results) is not int
        or state.rated_results < 0
        or state.rated_results > state.processed_sequence
        or not isinstance(state.processed_record_hash, str)
        or _HEX_64.fullmatch(state.processed_record_hash) is None
        or (
            state.processed_sequence == 0
            and state.processed_record_hash != ZERO_TEAM_HISTORY_HASH
        )
        or (
            state.processed_sequence > 0
            and state.processed_record_hash == ZERO_TEAM_HISTORY_HASH
        )
    ):
        _fail("team_rating_prefix_invalid", "Team rating prefix is invalid")
    if not isinstance(state.teams, tuple):
        _fail("team_rating_state_invalid", "Team rating mapping is invalid")
    team_ids: set[str] = set()
    for team in state.teams:
        if not isinstance(team, BlindTeamRating):
            _fail("team_rating_entry_invalid", "Team rating entry is invalid")
        _validate_team(team)
        if team.team_id in team_ids:
            _fail("team_rating_team_duplicate", "Team identity is duplicated")
        team_ids.add(team.team_id)
    if tuple(sorted(team_ids)) != tuple(team.team_id for team in state.teams):
        _fail("team_rating_team_order_invalid", "Team rating order is invalid")
    total_games = sum(team.games_played for team in state.teams)
    total_wins = sum(team.wins for team in state.teams)
    total_losses = sum(team.losses for team in state.teams)
    total_ties = sum(team.ties for team in state.teams)
    if (
        total_games != state.rated_results * 2
        or total_wins != total_losses
        or total_ties % 2 != 0
        or sum(team.rating for team in state.teams) != len(state.teams) * INITIAL_RATING
    ):
        _fail(
            "team_rating_counter_inconsistent", "Team rating counters are inconsistent"
        )
    if state.rated_results == 0 and state.teams:
        _fail("team_rating_mapping_inconsistent", "Empty rating history has entries")
    return state


def validate_team_battle_result(
    result: BlindTeamBattleResult,
) -> BlindTeamBattleResult:
    """Validate one neutral result without exposing either private identity."""

    if not isinstance(result, BlindTeamBattleResult):
        _fail("team_result_invalid", "Team battle result is invalid")
    if type(result.sequence) is not int or result.sequence <= 0:
        _fail("team_result_sequence_invalid", "Team result sequence is invalid")
    if (
        not isinstance(result.previous_record_hash, str)
        or _HEX_64.fullmatch(result.previous_record_hash) is None
        or not isinstance(result.record_hash, str)
        or _HEX_64.fullmatch(result.record_hash) is None
        or result.record_hash == ZERO_TEAM_HISTORY_HASH
        or result.record_hash == result.previous_record_hash
        or (
            result.sequence == 1
            and result.previous_record_hash != ZERO_TEAM_HISTORY_HASH
        )
        or (
            result.sequence > 1
            and result.previous_record_hash == ZERO_TEAM_HISTORY_HASH
        )
    ):
        _fail("team_result_hash_invalid", "Team result hash binding is invalid")
    if not is_valid_private_team_id(result.team_a_id) or not is_valid_private_team_id(
        result.team_b_id
    ):
        _fail("team_result_team_id_invalid", "Team result identity is invalid")
    if result.team_a_id == result.team_b_id:
        _fail("team_result_same_team", "Team result requires two distinct teams")
    if (
        not isinstance(result.outcome, str)
        or result.outcome not in TEAM_RESULT_OUTCOMES
    ):
        _fail("team_result_outcome_invalid", "Team result outcome is invalid")
    return result


def _updated_streak(team: BlindTeamRating, outcome: str) -> tuple[str, int]:
    if outcome == TEAM_OUTCOME_TIE:
        return STREAK_NONE, 0
    kind = STREAK_WIN if outcome == TEAM_OUTCOME_A_WIN else STREAK_LOSS
    length = team.streak_length + 1 if team.streak_kind == kind else 1
    return kind, length


def _opposite_outcome(outcome: str) -> str:
    if outcome == TEAM_OUTCOME_A_WIN:
        return TEAM_OUTCOME_A_LOSS
    if outcome == TEAM_OUTCOME_A_LOSS:
        return TEAM_OUTCOME_A_WIN
    return outcome


def _updated_team(
    team: BlindTeamRating,
    *,
    delta: int,
    outcome: str,
) -> BlindTeamRating:
    rating = team.rating + delta
    streak_kind, streak_length = _updated_streak(team, outcome)
    return BlindTeamRating(
        team.team_id,
        rating,
        team.games_played + 1,
        team.wins + int(outcome == TEAM_OUTCOME_A_WIN),
        team.losses + int(outcome == TEAM_OUTCOME_A_LOSS),
        team.ties + int(outcome == TEAM_OUTCOME_TIE),
        max(team.peak_rating, rating, INITIAL_RATING),
        streak_kind,
        streak_length,
    )


def apply_team_battle_result(
    state: BlindTeamRatingState,
    result: BlindTeamBattleResult,
) -> tuple[BlindTeamRatingState, BlindTeamRatingUpdate | None]:
    """Apply exactly the next neutral result to both team peers."""

    validate_team_rating_state(state)
    validate_team_battle_result(result)
    if (
        result.sequence != state.processed_sequence + 1
        or result.previous_record_hash != state.processed_record_hash
    ):
        _fail("team_result_sequence_invalid", "Team result order is invalid")
    if result.outcome == TEAM_OUTCOME_NO_RESULT:
        updated = BlindTeamRatingState(
            state.schema_version,
            state.algorithm_id,
            state.initial_rating,
            state.k_factor,
            state.scale,
            state.rounding_policy,
            result.sequence,
            result.record_hash,
            state.rated_results,
            state.teams,
        )
        return validate_team_rating_state(updated), None

    teams = {team.team_id: team for team in state.teams}
    team_a = teams.get(result.team_a_id, baseline_team_rating(result.team_a_id))
    team_b = teams.get(result.team_b_id, baseline_team_rating(result.team_b_id))
    delta_a = calculate_team_elo_delta(team_a.rating, team_b.rating, result.outcome)
    delta_b = -delta_a
    updated_a = _updated_team(team_a, delta=delta_a, outcome=result.outcome)
    updated_b = _updated_team(
        team_b,
        delta=delta_b,
        outcome=_opposite_outcome(result.outcome),
    )
    teams[result.team_a_id] = updated_a
    teams[result.team_b_id] = updated_b
    updated_state = BlindTeamRatingState(
        state.schema_version,
        state.algorithm_id,
        state.initial_rating,
        state.k_factor,
        state.scale,
        state.rounding_policy,
        result.sequence,
        result.record_hash,
        state.rated_results + 1,
        tuple(teams[team_id] for team_id in sorted(teams)),
    )
    validate_team_rating_state(updated_state)
    return updated_state, BlindTeamRatingUpdate(
        team_a.rating,
        updated_a.rating,
        delta_a,
        team_b.rating,
        updated_b.rating,
        delta_b,
    )


def derive_team_rating_state(
    results: Sequence[BlindTeamBattleResult],
) -> BlindTeamRatingState:
    """Replay an ordered neutral result history into a deterministic team state."""

    if not isinstance(results, Sequence) or isinstance(
        results, (str, bytes, bytearray)
    ):
        _fail("team_rating_history_invalid", "Team result history is invalid")
    state = empty_team_rating_state()
    for result in results:
        state, _update = apply_team_battle_result(state, result)
    return validate_team_rating_state(state)
