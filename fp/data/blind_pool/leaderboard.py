"""Public-safe identity registry and leaderboard projection for team Elo."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import unicodedata
from typing import Any, Mapping, NoReturn, Sequence

from .errors import BlindPoolValidationError
from .team_rating import (
    BlindTeamRatingState,
    baseline_team_rating,
    is_valid_private_team_id,
    validate_team_rating_state,
)


MAX_PUBLIC_TEAM_NAME_LENGTH = 40
PUBLIC_TEAM_KIND_PLAYER = "player"
PUBLIC_TEAM_KIND_BOT = "bot"
PUBLIC_TEAM_KINDS = frozenset({PUBLIC_TEAM_KIND_PLAYER, PUBLIC_TEAM_KIND_BOT})
_UNSAFE_MARKUP_CHARACTERS = frozenset("<>&")
_RESERVED_BOT_NAME_PREFIX = "bot team"


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _normalized_public_name_key(display_name: str) -> str:
    return unicodedata.normalize("NFC", display_name).casefold()


def validate_public_team_name(display_name: object) -> str:
    """Validate one normalized, display-only public team name."""

    if not isinstance(display_name, str):
        _fail("team_public_name_invalid", "Public team name is invalid")
    try:
        display_name.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _fail("team_public_name_invalid", "Public team name is invalid")
    if (
        not display_name
        or len(display_name) > MAX_PUBLIC_TEAM_NAME_LENGTH
        or display_name != display_name.strip()
        or display_name != unicodedata.normalize("NFC", display_name)
        or "  " in display_name
        or any(character.isspace() and character != " " for character in display_name)
        or any(not character.isprintable() for character in display_name)
        or any(character in _UNSAFE_MARKUP_CHARACTERS for character in display_name)
    ):
        _fail("team_public_name_invalid", "Public team name is invalid")
    return display_name


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamPublicIdentity:
    """Private stable identity paired with an intentionally public alias."""

    team_id: str = field(repr=False)
    display_name: str
    kind: str

    def __post_init__(self) -> None:
        if not is_valid_private_team_id(self.team_id):
            _fail("team_public_id_invalid", "Public registry team identity is invalid")
        validate_public_team_name(self.display_name)
        if not isinstance(self.kind, str) or self.kind not in PUBLIC_TEAM_KINDS:
            _fail("team_public_kind_invalid", "Public team kind is invalid")
        if self.kind == PUBLIC_TEAM_KIND_PLAYER and _normalized_public_name_key(
            self.display_name
        ).startswith(_RESERVED_BOT_NAME_PREFIX):
            _fail(
                "team_public_name_reserved",
                "Public team name uses a reserved namespace",
            )

    def __repr__(self) -> str:
        return "BlindTeamPublicIdentity(display_name={!r}, kind={!r})".format(
            self.display_name,
            self.kind,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindTeamPublicIdentity serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamPublicRegistry:
    """Immutable in-memory mapping boundary for public leaderboard identities."""

    identities: tuple[BlindTeamPublicIdentity, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identities, tuple):
            _fail("team_public_registry_invalid", "Public team registry is invalid")
        private_ids: set[str] = set()
        public_names: set[str] = set()
        for identity in self.identities:
            if not isinstance(identity, BlindTeamPublicIdentity):
                _fail("team_public_registry_invalid", "Public team registry is invalid")
            if identity.team_id in private_ids:
                _fail("team_public_id_duplicate", "Private team identity is duplicated")
            name_key = _normalized_public_name_key(identity.display_name)
            if name_key in public_names:
                _fail("team_public_name_duplicate", "Public team name is duplicated")
            private_ids.add(identity.team_id)
            public_names.add(name_key)

    @property
    def team_count(self) -> int:
        return len(self.identities)

    def identity(self, team_id: str) -> BlindTeamPublicIdentity | None:
        if not is_valid_private_team_id(team_id):
            _fail("team_public_id_invalid", "Public registry team identity is invalid")
        return next(
            (identity for identity in self.identities if identity.team_id == team_id),
            None,
        )

    def __repr__(self) -> str:
        return "BlindTeamPublicRegistry(team_count={!r})".format(self.team_count)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindTeamPublicRegistry serialization is disabled")


@dataclass(frozen=True, slots=True)
class BlindLeaderboardRow:
    """One public-only row with no stable private identity field."""

    rank: int
    name: str
    kind: str
    rating: int
    games: int
    wins: int
    losses: int
    ties: int
    win_rate: Decimal
    peak_rating: int
    streak: str

    @property
    def win_rate_display(self) -> str:
        rounded = self.win_rate.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        return "{:.1f}%".format(rounded)

    def to_public_dict(self) -> Mapping[str, Any]:
        """Return deterministic UI-oriented fields containing public data only."""

        return {
            "rank": self.rank,
            "name": self.name,
            "kind": self.kind,
            "rating": self.rating,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": self.win_rate_display,
            "peak_rating": self.peak_rating,
            "streak": self.streak,
        }


def _coerce_registry(
    identities: BlindTeamPublicRegistry | Sequence[BlindTeamPublicIdentity],
) -> BlindTeamPublicRegistry:
    if isinstance(identities, BlindTeamPublicRegistry):
        return identities
    if not isinstance(identities, Sequence) or isinstance(
        identities, (str, bytes, bytearray)
    ):
        _fail("team_public_registry_invalid", "Public team registry is invalid")
    return BlindTeamPublicRegistry(tuple(identities))


def build_leaderboard(
    team_rating_state: BlindTeamRatingState,
    registered_public_identities: (
        BlindTeamPublicRegistry | Sequence[BlindTeamPublicIdentity]
    ),
) -> tuple[BlindLeaderboardRow, ...]:
    """Project a complete, deterministic leaderboard without mutating state."""

    validate_team_rating_state(team_rating_state)
    registry = _coerce_registry(registered_public_identities)
    registered_ids = {identity.team_id for identity in registry.identities}
    if any(team.team_id not in registered_ids for team in team_rating_state.teams):
        _fail(
            "leaderboard_identity_missing",
            "A rated team has no registered public identity",
        )

    sortable = []
    for identity in registry.identities:
        rating = team_rating_state.team(identity.team_id)
        if rating is None:
            rating = baseline_team_rating(identity.team_id)
        sortable.append((identity, rating))
    sortable.sort(
        key=lambda item: (
            -item[1].rating,
            -item[1].games_played,
            -item[1].wins,
            _normalized_public_name_key(item[0].display_name),
            item[0].display_name,
        )
    )

    return tuple(
        BlindLeaderboardRow(
            rank,
            identity.display_name,
            identity.kind,
            rating.rating,
            rating.games_played,
            rating.wins,
            rating.losses,
            rating.ties,
            rating.win_rate_percent,
            rating.peak_rating,
            rating.streak_label,
        )
        for rank, (identity, rating) in enumerate(sortable, start=1)
    )
