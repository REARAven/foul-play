"""Immutable records and indexes for schema-version-1 complete team pools.

Roster identity uses exact species/form IDs, and one roster can map to multiple
team variants.  These records deliberately contain no battle objects or
hidden-information policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping

from fp.battle.helpers import normalize_name

from .errors import TeamPoolRegistryError


SCHEMA_VERSION = 1
NO_ITEM_ID = "none"
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
PUBLIC_FIELD_NAMES = frozenset(
    {
        "species_id",
        "base_species_id",
        "item_id",
        "base_ability_id",
        "current_ability_id",
        "move_ids",
        "nature_id",
        "evs",
        "ivs",
        "level",
    }
)

RosterKey = tuple[str, str, str, str, str, str]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty canonical normalized ID")
    if normalize_name(value) != value:
        raise ValueError(f"{label} must be a nonempty canonical normalized ID")
    return value


def canonical_roster_key(species_ids: Iterable[str]) -> RosterKey:
    """Return the sorted, immutable exact-form identity for one full roster."""

    values = tuple(species_ids)
    if len(values) != 6:
        raise ValueError("a canonical roster key requires exactly six species IDs")
    for species_id in values:
        _canonical_identifier(species_id, "species ID")
    return tuple(sorted(values))  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class PoolIdentity:
    """Stable identity of a versioned, format-scoped pool."""

    pool_id: str
    pool_version: str
    format_id: str


@dataclass(frozen=True, order=True)
class TeamRecordId:
    """Stable identity of one team variant within a pool."""

    team_id: str
    variant_id: str


@dataclass(frozen=True)
class SourceLocation:
    """Source filename and JSON path retained for diagnostics."""

    source: str
    path: str


@dataclass(frozen=True)
class StatValues:
    """Immutable values for the six standard stats."""

    hp: int
    atk: int
    defense: int
    spa: int
    spd: int
    spe: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.hp, self.atk, self.defense, self.spa, self.spd, self.spe)


@dataclass(frozen=True)
class PokemonRecord:
    """One immutable Pokémon entry in its submitted, initial state."""

    slot_id: str
    species_id: str
    base_species_id: str
    item_id: str
    base_ability_id: str
    current_ability_id: str
    move_ids: tuple[str, str, str, str]
    nature_id: str
    evs: StatValues
    ivs: StatValues
    level: int
    public_fields: tuple[str, ...]
    metadata: Mapping[str, Any]
    source_location: SourceLocation

    def __post_init__(self) -> None:
        object.__setattr__(self, "move_ids", tuple(self.move_ids))
        object.__setattr__(self, "public_fields", tuple(self.public_fields))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True)
class TeamRecord:
    """One immutable complete six-Pokémon team or variant."""

    record_id: TeamRecordId
    variant_of: TeamRecordId | None
    display_name: str | None
    pokemon: tuple[PokemonRecord, PokemonRecord, PokemonRecord, PokemonRecord, PokemonRecord, PokemonRecord]
    roster_key: RosterKey
    metadata: Mapping[str, Any]
    source_location: SourceLocation

    def __post_init__(self) -> None:
        object.__setattr__(self, "pokemon", tuple(self.pokemon))
        object.__setattr__(self, "roster_key", tuple(self.roster_key))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True)
class TeamPool:
    """An immutable pool with deterministic ID and exact-roster indexes."""

    schema_version: int
    identity: PoolIdentity
    patch_version: str | None
    source_documents: tuple[str, ...]
    display_name: str | None
    default_public_fields: tuple[str, ...]
    metadata: Mapping[str, Any]
    teams: tuple[TeamRecord, ...]
    _team_lookup: Mapping[TeamRecordId, TeamRecord] = field(init=False, repr=False)
    _roster_index: Mapping[RosterKey, tuple[TeamRecordId, ...]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        teams = tuple(sorted(self.teams, key=lambda team: team.record_id))
        lookup = {team.record_id: team for team in teams}
        index_lists: dict[RosterKey, list[TeamRecordId]] = {}
        for team in teams:
            index_lists.setdefault(team.roster_key, []).append(team.record_id)
        index = {
            key: tuple(sorted(record_ids))
            for key, record_ids in sorted(index_lists.items())
        }
        object.__setattr__(self, "source_documents", tuple(self.source_documents))
        object.__setattr__(self, "default_public_fields", tuple(self.default_public_fields))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(self, "teams", teams)
        object.__setattr__(self, "_team_lookup", MappingProxyType(lookup))
        object.__setattr__(self, "_roster_index", MappingProxyType(index))

    @property
    def team_lookup(self) -> Mapping[TeamRecordId, TeamRecord]:
        return self._team_lookup

    @property
    def roster_index(self) -> Mapping[RosterKey, tuple[TeamRecordId, ...]]:
        return self._roster_index

    def get_team(self, record_id: TeamRecordId) -> TeamRecord | None:
        return self._team_lookup.get(record_id)

    def matching_team_ids(self, roster_key: RosterKey) -> tuple[TeamRecordId, ...]:
        return self._roster_index.get(roster_key, ())


@dataclass(frozen=True)
class TeamPoolRegistry:
    """Immutable collection of pools keyed by versioned format identity.

    Duplicate identities are always rejected, including byte-identical loads.
    """

    pools: tuple[TeamPool, ...] = ()
    _pool_lookup: Mapping[PoolIdentity, TeamPool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pools = tuple(sorted(self.pools, key=lambda pool: pool.identity))
        lookup: dict[PoolIdentity, TeamPool] = {}
        for pool in pools:
            if pool.identity in lookup:
                raise TeamPoolRegistryError(
                    f"duplicate pool identity: {pool.identity!r}; duplicates are always rejected"
                )
            lookup[pool.identity] = pool
        object.__setattr__(self, "pools", pools)
        object.__setattr__(self, "_pool_lookup", MappingProxyType(lookup))

    def __iter__(self) -> Iterator[TeamPool]:
        return iter(self.pools)

    def __len__(self) -> int:
        return len(self.pools)

    def get(self, identity: PoolIdentity) -> TeamPool | None:
        return self._pool_lookup.get(identity)

    @property
    def pool_lookup(self) -> Mapping[PoolIdentity, TeamPool]:
        return self._pool_lookup
