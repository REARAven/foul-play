"""Immutable, privacy-safe records for external Blind Ladder state."""

from __future__ import annotations

from dataclasses import dataclass, field
import hmac
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator, Mapping, NoReturn

from .errors import BlindPoolValidationError

if TYPE_CHECKING:
    from .leaderboard import BlindTeamPublicIdentity


SCHEMA_VERSION = 1
SUPPORTED_FORMAT_ID = "gen9tugs"
OPAQUE_TEAM_ID_PATTERN = re.compile(r"^BL-[0-9]{3,}-v[1-9][0-9]*$")
CHALLENGE_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PLAYER_TEAM_ID_PATTERN = re.compile(r"^player-team:[0-9a-f]{32}$")


def is_valid_opaque_team_id(value: object) -> bool:
    """Return whether a value satisfies the shared opaque Blind Ladder ID contract."""

    return (
        isinstance(value, str) and OPAQUE_TEAM_ID_PATTERN.fullmatch(value) is not None
    )


def is_valid_player_team_id(value: object) -> bool:
    """Return whether a value satisfies the private player-team ID contract."""

    return (
        isinstance(value, str) and PLAYER_TEAM_ID_PATTERN.fullmatch(value) is not None
    )


class BlindChallengeToken:
    """One immutable exact server-issued challenge correlation token."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if (
            not isinstance(value, str)
            or CHALLENGE_TOKEN_PATTERN.fullmatch(value) is None
        ):
            raise BlindPoolValidationError(
                "challenge_token_invalid",
                "Blind Ladder challenge token is malformed",
            ) from None
        object.__setattr__(self, "_value", value)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("BlindChallengeToken is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("BlindChallengeToken is immutable")

    def __repr__(self) -> str:
        return "BlindChallengeToken(configured=True)"

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindChallengeToken serialization is disabled")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlindChallengeToken):
            return NotImplemented
        return hmac.compare_digest(self._value, other._value)

    def matches(self, other: BlindChallengeToken) -> bool:
        """Compare two already validated tokens without data-dependent timing."""

        return isinstance(other, BlindChallengeToken) and hmac.compare_digest(
            self._value,
            other._value,
        )

    def wire_value(self) -> str:
        """Expose the exact value only at a private protocol/state boundary."""

        return self._value


@dataclass(frozen=True, repr=False)
class BlindPoolConfig:
    """Explicit locations for private files kept outside the repository."""

    private_root: Path
    registry_path: Path

    def __post_init__(self) -> None:
        private_root = Path(self.private_root)
        registry_path = Path(self.registry_path)
        if not private_root.is_absolute() or not registry_path.is_absolute():
            raise BlindPoolValidationError(
                "config_path_invalid",
                "Blind Ladder private root and registry paths must be absolute",
            )
        object.__setattr__(self, "private_root", private_root)
        object.__setattr__(self, "registry_path", registry_path)

    def __repr__(self) -> str:
        return "BlindPoolConfig(configured=True)"


@dataclass(frozen=True, repr=False)
class BlindPoolStateConfig:
    """External location of persistent shuffled-bag bookkeeping."""

    pool_config: BlindPoolConfig
    state_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.pool_config, BlindPoolConfig):
            raise BlindPoolValidationError(
                "state_config_invalid",
                "Blind Ladder state configuration has an invalid type",
            )
        try:
            state_path = Path(self.state_path)
        except TypeError:
            raise BlindPoolValidationError(
                "state_config_path_invalid",
                "Blind Ladder state path must be absolute",
            ) from None
        if not state_path.is_absolute():
            raise BlindPoolValidationError(
                "state_config_path_invalid",
                "Blind Ladder state path must be absolute",
            )
        object.__setattr__(self, "state_path", state_path)

    @property
    def lock_path(self) -> Path:
        return self.state_path.with_name(self.state_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindPoolStateConfig(configured=True)"


@dataclass(frozen=True, repr=False)
class BlindPoolEntry:
    """One opaque registry entry and its verified external team file."""

    team_id: str
    active: bool
    relative_team_path: str
    resolved_team_path: Path
    sha256: str

    def __repr__(self) -> str:
        return "BlindPoolEntry(active={!r})".format(self.active)


@dataclass(frozen=True, repr=False)
class BlindPoolRegistry:
    """A validated registry with deterministic opaque-ID lookup."""

    schema_version: int
    registry_version: str
    format_id: str
    entries: tuple[BlindPoolEntry, ...]
    _entry_by_id: Mapping[str, BlindPoolEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        entry_by_id = {entry.team_id: entry for entry in entries}
        if len(entry_by_id) != len(entries):
            raise BlindPoolValidationError(
                "duplicate_team_id",
                "Blind Ladder registry contains duplicate opaque team IDs",
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_entry_by_id", MappingProxyType(entry_by_id))

    def __iter__(self) -> Iterator[BlindPoolEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return "BlindPoolRegistry(entry_count={!r}, active_count={!r})".format(
            len(self.entries),
            len(self.active_entries),
        )

    @property
    def active_entries(self) -> tuple[BlindPoolEntry, ...]:
        return tuple(entry for entry in self.entries if entry.active)

    def get_entry(self, team_id: str) -> BlindPoolEntry | None:
        return self._entry_by_id.get(team_id)


@dataclass(frozen=True, repr=False)
class BlindPoolReservation:
    """One unresolved bag reservation without private team material."""

    reservation_id: str
    team_id: str
    cycle_number: int
    position: int
    phase: str
    challenge_token: BlindChallengeToken | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            "BlindPoolReservation(cycle_number={!r}, position={!r}, phase={!r})"
        ).format(self.cycle_number, self.position, self.phase)


@dataclass(frozen=True, repr=False)
class BlindPoolBagState:
    """Validated persistent shuffled-bag state."""

    schema_version: int
    registry_fingerprint: str
    cycle_number: int
    cycle_order: tuple[str, ...]
    next_index: int
    last_consumed_id: str | None
    reservation: BlindPoolReservation | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_order", tuple(self.cycle_order))

    def __repr__(self) -> str:
        return (
            "BlindPoolBagState(cycle_number={!r}, entry_count={!r}, "
            "next_index={!r}, reservation_pending={!r})"
        ).format(
            self.cycle_number,
            len(self.cycle_order),
            self.next_index,
            self.reservation is not None,
        )


@dataclass(frozen=True, repr=False)
class BlindPoolChallenge:
    """Minimal process-local identity for one supported incoming challenge."""

    challenger_id: str
    challenger_name: str
    format_id: str
    source: str
    challenge_token: BlindChallengeToken | None = field(default=None, repr=False)
    player_team_identity: BlindTeamPublicIdentity | None = field(
        default=None, repr=False
    )

    @property
    def deduplication_identity(self) -> BlindChallengeToken | tuple[str, str]:
        """Use exact token identity when present and legacy PM identity otherwise."""

        if self.challenge_token is not None:
            return self.challenge_token
        return self.challenger_id, self.format_id

    def __repr__(self) -> str:
        return "BlindPoolChallenge(format_id={!r}, source={!r})".format(
            self.format_id,
            self.source,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindPoolChallenge serialization is disabled")


@dataclass(frozen=True, repr=False)
class BlindChallengeTeamIdentity:
    """One private player-team identity correlated to an exact challenge token."""

    challenge_token: BlindChallengeToken = field(repr=False)
    public_identity: BlindTeamPublicIdentity = field(repr=False)

    def __repr__(self) -> str:
        display_name = getattr(self.public_identity, "display_name", None)
        return "BlindChallengeTeamIdentity(display_name={!r})".format(display_name)

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindChallengeTeamIdentity serialization is disabled")


@dataclass(frozen=True, repr=False)
class BlindChallengeEndEvent:
    """Private invalidation for one exact challenge token."""

    challenge_token: BlindChallengeToken = field(repr=False)

    def __repr__(self) -> str:
        return "BlindChallengeEndEvent()"

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindChallengeEndEvent serialization is disabled")


@dataclass(frozen=True, repr=False)
class BlindChallengeRoomBinding:
    """Private exact-token binding to one structurally valid battle room."""

    challenge_token: BlindChallengeToken = field(repr=False)
    room_id: str

    def __repr__(self) -> str:
        return "BlindChallengeRoomBinding(room_id={!r})".format(self.room_id)

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindChallengeRoomBinding serialization is disabled")


@dataclass(frozen=True, repr=False)
class BlindPoolBattleRoom:
    """Exactly correlated battle-room metadata without retained raw traffic."""

    room_id: str
    format_id: str
    opponent_id: str
    opponent_name: str
    bot_slot: str
    opponent_slot: str

    def __repr__(self) -> str:
        return "BlindPoolBattleRoom(room_id={!r}, format_id={!r})".format(
            self.room_id,
            self.format_id,
        )
