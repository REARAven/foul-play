"""Deterministic Elo-v1 projection derived from Blind Ladder results."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re
from typing import Any, NoReturn, Sequence

from .errors import BlindPoolValidationError
from .models import is_valid_opaque_team_id
from .result_ledger import (
    OUTCOME_NO_RESULT,
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
    BlindCompletedBattleResult,
)


RATING_STATE_SCHEMA_VERSION = 1
RATING_ALGORITHM_ID = "elo-v1"
INITIAL_RATING = 1500
K_FACTOR = 32
RATING_SCALE = 400
ROUNDING_POLICY = "half-away-from-zero"
RATING_STATE_PATH_ENV = "TUGS_BLIND_RATING_STATE"
ZERO_HISTORY_HASH = "0" * 64

STREAK_NONE = "none"
STREAK_WIN = "win"
STREAK_LOSS = "loss"
STREAK_KINDS = frozenset({STREAK_NONE, STREAK_WIN, STREAK_LOSS})

_SHOWDOWN_ID = re.compile(r"^[a-z0-9]{1,18}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RATED_OUTCOMES = frozenset({OUTCOME_PLAYER_WIN, OUTCOME_PLAYER_LOSS, OUTCOME_TIE})
_SCORES = {
    OUTCOME_PLAYER_WIN: Decimal("1"),
    OUTCOME_PLAYER_LOSS: Decimal("0"),
    OUTCOME_TIE: Decimal("0.5"),
}


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _validate_integer(value: Any, *, context: str) -> int:
    if type(value) is not int:
        _fail("rating_integer_invalid", "{} must be an integer".format(context))
    return value


def round_half_away_from_zero(value: Decimal) -> int:
    """Round a finite Decimal to the nearest integer, with halves away from zero."""

    if not isinstance(value, Decimal) or not value.is_finite():
        _fail("rating_rounding_input_invalid", "Elo rounding input is invalid")
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def expected_player_score(player_rating: int, opponent_rating: int) -> Decimal:
    """Return the deterministic high-precision Elo expected player score."""

    player = _validate_integer(player_rating, context="Player rating")
    opponent = _validate_integer(opponent_rating, context="Opponent rating")
    with localcontext() as context:
        context.prec = 50
        exponent = Decimal(opponent - player) / Decimal(RATING_SCALE)
        return Decimal(1) / (Decimal(1) + (Decimal(10) ** exponent))


def calculate_elo_delta(
    player_rating: int,
    opponent_rating: int,
    outcome: str,
) -> int:
    """Return the integer Elo-v1 player delta for one rated outcome."""

    if not isinstance(outcome, str) or outcome not in _RATED_OUTCOMES:
        _fail("rating_outcome_invalid", "Elo outcome is not rateable")
    expected = expected_player_score(player_rating, opponent_rating)
    with localcontext() as context:
        context.prec = 50
        raw_delta = Decimal(K_FACTOR) * (_SCORES[outcome] - expected)
        return round_half_away_from_zero(raw_delta)


@dataclass(frozen=True, slots=True, repr=False)
class BlindPlayerRating:
    player_id: str = field(repr=False)
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

    def __repr__(self) -> str:
        return (
            "BlindPlayerRating(rating={!r}, games_played={!r}, "
            "wins={!r}, losses={!r}, ties={!r})"
        ).format(
            self.rating,
            self.games_played,
            self.wins,
            self.losses,
            self.ties,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindPlayerRating serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class BlindOpponentRating:
    team_id: str = field(repr=False)
    rating: int = field(repr=False)
    games_played: int = field(repr=False)

    def __repr__(self) -> str:
        return "BlindOpponentRating(private=True)"

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindOpponentRating serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class BlindRatingState:
    schema_version: int
    algorithm_id: str
    initial_rating: int
    k_factor: int
    scale: int
    rounding_policy: str
    processed_sequence: int
    processed_record_hash: str = field(repr=False)
    rated_results: int
    players: tuple[BlindPlayerRating, ...] = field(repr=False)
    opponents: tuple[BlindOpponentRating, ...] = field(repr=False)

    @property
    def processed_results(self) -> int:
        return self.processed_sequence

    @property
    def player_count(self) -> int:
        return len(self.players)

    def player(self, player_id: str) -> BlindPlayerRating | None:
        return next(
            (player for player in self.players if player.player_id == player_id),
            None,
        )

    def __repr__(self) -> str:
        return (
            "BlindRatingState(processed_results={!r}, rated_results={!r}, "
            "player_count={!r})"
        ).format(
            self.processed_results,
            self.rated_results,
            self.player_count,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindRatingState serialization is disabled")


@dataclass(frozen=True, slots=True)
class BlindRatingUpdate:
    rating_before: int
    rating_after: int
    rating_delta: int
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


def empty_rating_state() -> BlindRatingState:
    return BlindRatingState(
        RATING_STATE_SCHEMA_VERSION,
        RATING_ALGORITHM_ID,
        INITIAL_RATING,
        K_FACTOR,
        RATING_SCALE,
        ROUNDING_POLICY,
        0,
        ZERO_HISTORY_HASH,
        0,
        (),
        (),
    )


def baseline_player_rating(player_id: str) -> BlindPlayerRating:
    if not isinstance(player_id, str) or _SHOWDOWN_ID.fullmatch(player_id) is None:
        _fail("rating_player_id_invalid", "Player identity is invalid")
    return BlindPlayerRating(
        player_id,
        INITIAL_RATING,
        0,
        0,
        0,
        0,
        INITIAL_RATING,
        STREAK_NONE,
        0,
    )


def _validate_player(player: BlindPlayerRating) -> None:
    if (
        not isinstance(player.player_id, str)
        or _SHOWDOWN_ID.fullmatch(player.player_id) is None
    ):
        _fail("rating_player_id_invalid", "Player identity is invalid")
    integers = (
        player.rating,
        player.games_played,
        player.wins,
        player.losses,
        player.ties,
        player.peak_rating,
        player.streak_length,
    )
    if any(type(value) is not int for value in integers):
        _fail("rating_player_invalid", "Player rating state is invalid")
    if (
        player.games_played <= 0
        or min(player.wins, player.losses, player.ties, player.streak_length) < 0
        or player.games_played != player.wins + player.losses + player.ties
        or player.peak_rating < INITIAL_RATING
        or player.peak_rating < player.rating
        or player.streak_kind not in STREAK_KINDS
        or (player.streak_kind == STREAK_NONE and player.streak_length != 0)
        or (player.streak_kind != STREAK_NONE and player.streak_length <= 0)
        or player.streak_length > player.games_played
        or (player.streak_kind == STREAK_WIN and player.streak_length > player.wins)
        or (player.streak_kind == STREAK_LOSS and player.streak_length > player.losses)
    ):
        _fail("rating_player_invalid", "Player rating state is inconsistent")


def _validate_opponent(opponent: BlindOpponentRating) -> None:
    if not is_valid_opaque_team_id(opponent.team_id):
        _fail("rating_opponent_id_invalid", "Opponent identity is invalid")
    if (
        type(opponent.rating) is not int
        or type(opponent.games_played) is not int
        or opponent.games_played <= 0
    ):
        _fail("rating_opponent_invalid", "Opponent rating state is invalid")


def validate_rating_state(state: BlindRatingState) -> BlindRatingState:
    """Validate one immutable in-memory Elo-v1 projection."""

    if not isinstance(state, BlindRatingState):
        _fail("rating_state_invalid", "Rating state is invalid")
    if (
        type(state.schema_version) is not int
        or state.schema_version != RATING_STATE_SCHEMA_VERSION
        or state.algorithm_id != RATING_ALGORITHM_ID
        or type(state.initial_rating) is not int
        or state.initial_rating != INITIAL_RATING
        or type(state.k_factor) is not int
        or state.k_factor != K_FACTOR
        or type(state.scale) is not int
        or state.scale != RATING_SCALE
        or state.rounding_policy != ROUNDING_POLICY
    ):
        _fail("rating_algorithm_unsupported", "Rating algorithm is unsupported")
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
            and state.processed_record_hash != ZERO_HISTORY_HASH
        )
        or (
            state.processed_sequence > 0
            and state.processed_record_hash == ZERO_HISTORY_HASH
        )
    ):
        _fail("rating_processed_prefix_invalid", "Rating ledger prefix is invalid")
    if not isinstance(state.players, tuple) or not isinstance(state.opponents, tuple):
        _fail("rating_state_invalid", "Rating mappings are invalid")
    player_ids: set[str] = set()
    opponent_ids: set[str] = set()
    for player in state.players:
        if not isinstance(player, BlindPlayerRating):
            _fail("rating_player_invalid", "Player rating entry is invalid")
        _validate_player(player)
        if player.player_id in player_ids:
            _fail("rating_player_duplicate", "Player rating identity is duplicated")
        player_ids.add(player.player_id)
    for opponent in state.opponents:
        if not isinstance(opponent, BlindOpponentRating):
            _fail("rating_opponent_invalid", "Opponent rating entry is invalid")
        _validate_opponent(opponent)
        if opponent.team_id in opponent_ids:
            _fail("rating_opponent_duplicate", "Opponent rating identity is duplicated")
        opponent_ids.add(opponent.team_id)
    if tuple(sorted(player_ids)) != tuple(player.player_id for player in state.players):
        _fail("rating_player_order_invalid", "Player rating order is invalid")
    if tuple(sorted(opponent_ids)) != tuple(
        opponent.team_id for opponent in state.opponents
    ):
        _fail("rating_opponent_order_invalid", "Opponent rating order is invalid")
    if (
        sum(player.games_played for player in state.players) != state.rated_results
        or sum(opponent.games_played for opponent in state.opponents)
        != state.rated_results
    ):
        _fail("rating_counter_inconsistent", "Rating counters are inconsistent")
    if state.rated_results == 0 and (state.players or state.opponents):
        _fail("rating_mapping_inconsistent", "Empty rating history has entries")
    return state


def _updated_streak(player: BlindPlayerRating, outcome: str) -> tuple[str, int]:
    if outcome == OUTCOME_TIE:
        return STREAK_NONE, 0
    kind = STREAK_WIN if outcome == OUTCOME_PLAYER_WIN else STREAK_LOSS
    length = player.streak_length + 1 if player.streak_kind == kind else 1
    return kind, length


def apply_completed_result(
    state: BlindRatingState,
    record: BlindCompletedBattleResult,
) -> tuple[BlindRatingState, BlindRatingUpdate | None]:
    """Apply exactly the next authoritative completed result to a projection."""

    validate_rating_state(state)
    if not isinstance(record, BlindCompletedBattleResult):
        _fail("rating_record_invalid", "Completed result record is invalid")
    if (
        record.sequence != state.processed_sequence + 1
        or record.previous_record_hash != state.processed_record_hash
    ):
        _fail("rating_record_sequence_invalid", "Completed result order is invalid")
    if record.outcome == OUTCOME_NO_RESULT:
        updated = BlindRatingState(
            state.schema_version,
            state.algorithm_id,
            state.initial_rating,
            state.k_factor,
            state.scale,
            state.rounding_policy,
            record.sequence,
            record.record_hash,
            state.rated_results,
            state.players,
            state.opponents,
        )
        return validate_rating_state(updated), None
    if record.outcome not in _RATED_OUTCOMES:
        _fail("rating_outcome_invalid", "Completed result outcome is unsupported")

    players = {player.player_id: player for player in state.players}
    opponents = {opponent.team_id: opponent for opponent in state.opponents}
    player = players.get(record.player_id, baseline_player_rating(record.player_id))
    opponent = opponents.get(
        record.team_id,
        BlindOpponentRating(record.team_id, INITIAL_RATING, 0),
    )
    delta = calculate_elo_delta(player.rating, opponent.rating, record.outcome)
    new_rating = player.rating + delta
    streak_kind, streak_length = _updated_streak(player, record.outcome)
    wins = player.wins + int(record.outcome == OUTCOME_PLAYER_WIN)
    losses = player.losses + int(record.outcome == OUTCOME_PLAYER_LOSS)
    ties = player.ties + int(record.outcome == OUTCOME_TIE)
    updated_player = BlindPlayerRating(
        player.player_id,
        new_rating,
        player.games_played + 1,
        wins,
        losses,
        ties,
        max(player.peak_rating, new_rating, INITIAL_RATING),
        streak_kind,
        streak_length,
    )
    updated_opponent = BlindOpponentRating(
        opponent.team_id,
        opponent.rating - delta,
        opponent.games_played + 1,
    )
    players[player.player_id] = updated_player
    opponents[opponent.team_id] = updated_opponent
    updated_state = BlindRatingState(
        state.schema_version,
        state.algorithm_id,
        state.initial_rating,
        state.k_factor,
        state.scale,
        state.rounding_policy,
        record.sequence,
        record.record_hash,
        state.rated_results + 1,
        tuple(players[player_id] for player_id in sorted(players)),
        tuple(opponents[team_id] for team_id in sorted(opponents)),
    )
    validate_rating_state(updated_state)
    return updated_state, BlindRatingUpdate(
        player.rating,
        updated_player.rating,
        delta,
        updated_player.games_played,
        updated_player.wins,
        updated_player.losses,
        updated_player.ties,
        updated_player.peak_rating,
        updated_player.streak_kind,
        updated_player.streak_length,
    )


def derive_rating_state(
    completed_results: Sequence[BlindCompletedBattleResult],
) -> BlindRatingState:
    """Replay authoritative ledger sequence order into one deterministic state."""

    if not isinstance(completed_results, Sequence):
        _fail("rating_history_invalid", "Completed result history is invalid")
    state = empty_rating_state()
    for record in completed_results:
        state, _update = apply_completed_result(state, record)
    return validate_rating_state(state)
