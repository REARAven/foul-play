"""Immutable, privacy-safe records for an external Blind Ladder registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

from .errors import BlindPoolValidationError


SCHEMA_VERSION = 1
SUPPORTED_FORMAT_ID = "gen9tugs"


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
