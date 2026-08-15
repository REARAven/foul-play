"""Immutable, privacy-safe models for dormant canonical Blind Ladder artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

from .models import is_valid_opaque_team_id


CANONICAL_REGISTRY_SCHEMA_VERSION = 1
CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION = 1
CANONICAL_ARTIFACT_SCHEMA_VERSION = 1
CANONICAL_METADATA_SCHEMA_VERSION = 1
CANONICAL_SIDECAR_SCHEMA_VERSION = 1
CANONICAL_PROVISIONER_PROFILE_VERSION = 1
CANONICAL_FORMAT_ID = "gen9tugs"
CANONICAL_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_CANONICAL_CONSTRUCTION_TOKEN = object()


class CanonicalArtifactError(ValueError):
    """A sanitized canonical registry or artifact validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        team_id: str | None = None,
    ) -> None:
        self.code = code
        self.team_id = team_id
        context = "" if team_id is None else " team_id={}".format(team_id)
        super().__init__("{}: {}{}".format(code, message, context))


def _fail(
    code: str,
    message: str,
    *,
    team_id: str | None = None,
) -> NoReturn:
    raise CanonicalArtifactError(code, message, team_id=team_id) from None


class _DuplicateJsonFieldError(ValueError):
    pass


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError


def _strict_json_bytes(
    raw: bytes,
    *,
    encoding_code: str,
    duplicate_code: str,
    json_code: str,
    team_id: str | None = None,
) -> Any:
    """Decode strict UTF-8 JSON while preserving duplicate-key failures."""

    if not isinstance(raw, bytes):
        _fail(encoding_code, "Canonical JSON encoding is invalid", team_id=team_id)
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(encoding_code, "Canonical JSON encoding is invalid", team_id=team_id)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = None
    if text is None:
        _fail(encoding_code, "Canonical JSON encoding is invalid", team_id=team_id)
    if "\x00" in text or "\ufeff" in text:
        _fail(encoding_code, "Canonical JSON encoding is invalid", team_id=team_id)
    duplicate_field = False
    malformed_json = False
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJsonFieldError:
        duplicate_field = True
    except (json.JSONDecodeError, ValueError):
        malformed_json = True
    if duplicate_field:
        _fail(
            duplicate_code,
            "Canonical JSON contains a duplicate field",
            team_id=team_id,
        )
    if malformed_json:
        _fail(json_code, "Canonical JSON is malformed", team_id=team_id)
    return document


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    *,
    code: str,
    context: str,
    team_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code, "{} must be an object".format(context), team_id=team_id)
    if expected - value.keys():
        _fail(code, "{} is missing required fields".format(context), team_id=team_id)
    if value.keys() - expected:
        _fail(code, "{} contains unexpected fields".format(context), team_id=team_id)
    return value


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalRegistryEntry:
    """One immutable canonical-registry entry without an artifact path."""

    team_id: str
    active: bool
    metadata_sha256: str = field(repr=False)

    def __repr__(self) -> str:
        return "CanonicalRegistryEntry(team_id={!r}, active={!r})".format(
            self.team_id,
            self.active,
        )


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalArtifactMetadata:
    """Strictly parsed metadata whose ordinary representation omits provenance."""

    schema_version: int
    team_id: str
    format_id: str
    format_fingerprint_sha256: str = field(repr=False)
    validation_status: str
    team_size: int
    showdown_commit: str = field(repr=False)
    showdown_tree_clean: bool
    showdown_package_version: str = field(repr=False)
    package_lock_sha256: str = field(repr=False)
    dist_tree_sha256: str = field(repr=False)
    node_version: str = field(repr=False)
    npm_version: str = field(repr=False)
    provisioner_version: int
    provisioner_sha256: str = field(repr=False)
    source_sha256: str = field(repr=False)
    packed_sha256: str = field(repr=False)
    sidecar_sha256: str = field(repr=False)
    semantic_team_fingerprint_sha256: str = field(repr=False)

    @property
    def active_provenance(self) -> tuple[object, ...]:
        return (
            self.schema_version,
            self.showdown_commit,
            self.showdown_package_version,
            self.showdown_tree_clean,
            self.format_fingerprint_sha256,
            self.package_lock_sha256,
            self.dist_tree_sha256,
            self.node_version,
            self.npm_version,
            self.provisioner_version,
            self.provisioner_sha256,
        )

    def __repr__(self) -> str:
        return (
            "CanonicalArtifactMetadata(schema_version={!r}, team_id={!r}, "
            "format_id={!r}, validation_status={!r}, team_size={!r})"
        ).format(
            self.schema_version,
            self.team_id,
            self.format_id,
            self.validation_status,
            self.team_size,
        )


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalBattleTeamRecord:
    """The only sidecar fields retained for existing battle initialization."""

    species_id: str
    nature_id: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int]

    def __repr__(self) -> str:
        return "CanonicalBattleTeamRecord(private=True)"


@dataclass(frozen=True, slots=True, repr=False)
class ParsedCanonicalSidecar:
    """A minimal immutable result of full structural sidecar validation."""

    schema_version: int
    team_id: str
    format_id: str
    records: tuple[CanonicalBattleTeamRecord, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            "ParsedCanonicalSidecar(schema_version={!r}, team_id={!r}, "
            "format_id={!r}, set_count={!r})"
        ).format(
            self.schema_version,
            self.team_id,
            self.format_id,
            len(self.records),
        )


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalRuntimeRegistry:
    """A verified canonical registry bound to immutable metadata snapshots."""

    schema_version: int
    registry_version: str
    format_id: str
    artifact_schema_version: int
    metadata_schema_version: int
    entries: tuple[CanonicalRegistryEntry, ...]
    _private_root: Path = field(repr=False)
    _registry_path: Path = field(repr=False)
    _metadata: tuple[CanonicalArtifactMetadata, ...] = field(repr=False)
    _fingerprint: str = field(repr=False)
    _construction_token: object = field(repr=False)
    _entry_by_id: Mapping[str, CanonicalRegistryEntry] = field(
        init=False,
        repr=False,
    )
    _metadata_by_id: Mapping[str, CanonicalArtifactMetadata] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self._construction_token is not _CANONICAL_CONSTRUCTION_TOKEN:
            _fail(
                "CANONICAL_INTERNAL_ERROR",
                "Canonical registry construction is restricted",
            )
        entries = tuple(self.entries)
        metadata = tuple(self._metadata)
        entry_by_id = {entry.team_id: entry for entry in entries}
        metadata_by_id = {item.team_id: item for item in metadata}
        if (
            len(entry_by_id) != len(entries)
            or len(metadata_by_id) != len(metadata)
            or entry_by_id.keys() != metadata_by_id.keys()
        ):
            _fail(
                "CANONICAL_INTERNAL_ERROR",
                "Canonical registry construction failed",
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_entry_by_id", MappingProxyType(entry_by_id))
        object.__setattr__(self, "_metadata_by_id", MappingProxyType(metadata_by_id))

    @property
    def active_ids(self) -> tuple[str, ...]:
        return tuple(entry.team_id for entry in self.entries if entry.active)

    @property
    def registry_fingerprint(self) -> str:
        return self._fingerprint

    def get_entry(self, team_id: str) -> CanonicalRegistryEntry | None:
        return self._entry_by_id.get(team_id)

    def _metadata_for(self, team_id: str) -> CanonicalArtifactMetadata | None:
        return self._metadata_by_id.get(team_id)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            "CanonicalRuntimeRegistry(schema_version={!r}, registry_version={!r}, "
            "format_id={!r}, entry_count={!r}, active_count={!r})"
        ).format(
            self.schema_version,
            self.registry_version,
            self.format_id,
            len(self.entries),
            len(self.active_ids),
        )


class CanonicalTeamArtifact:
    """One verified selected team with deliberately narrow private accessors."""

    __slots__ = (
        "_artifact_schema_version",
        "_format_id",
        "_metadata_schema_version",
        "_packed_wire",
        "_records",
        "_sidecar_schema_version",
        "_team_id",
    )

    def __init__(
        self,
        *,
        team_id: str,
        format_id: str,
        artifact_schema_version: int,
        metadata_schema_version: int,
        sidecar_schema_version: int,
        packed_wire: str,
        records: tuple[CanonicalBattleTeamRecord, ...],
        construction_token: object,
    ) -> None:
        if construction_token is not _CANONICAL_CONSTRUCTION_TOKEN:
            _fail(
                "CANONICAL_INTERNAL_ERROR",
                "Canonical artifact construction is restricted",
                team_id=team_id if is_valid_opaque_team_id(team_id) else None,
            )
        object.__setattr__(self, "_team_id", team_id)
        object.__setattr__(self, "_format_id", format_id)
        object.__setattr__(self, "_artifact_schema_version", artifact_schema_version)
        object.__setattr__(self, "_metadata_schema_version", metadata_schema_version)
        object.__setattr__(self, "_sidecar_schema_version", sidecar_schema_version)
        object.__setattr__(self, "_packed_wire", packed_wire)
        object.__setattr__(self, "_records", tuple(records))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CanonicalTeamArtifact is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("CanonicalTeamArtifact is immutable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("CanonicalTeamArtifact serialization is disabled")

    @property
    def team_id(self) -> str:
        return self._team_id

    @property
    def format_id(self) -> str:
        return self._format_id

    @property
    def artifact_schema_version(self) -> int:
        return self._artifact_schema_version

    @property
    def metadata_schema_version(self) -> int:
        return self._metadata_schema_version

    @property
    def sidecar_schema_version(self) -> int:
        return self._sidecar_schema_version

    @property
    def set_count(self) -> int:
        return len(self._records)

    def packed_for_submission(self) -> str:
        """Return the exact verified Showdown wire value."""

        return self._packed_wire

    def new_battle_team_projection(self) -> list[dict[str, object]]:
        """Return a fresh legacy-compatible minimal own-team projection."""

        projection: list[dict[str, object]] = []
        for record in self._records:
            projection.append(
                {
                    "species": record.species_id,
                    "nature": record.nature_id,
                    "evs": {
                        stat: str(value)
                        for stat, value in zip(CANONICAL_STAT_KEYS, record.evs)
                    },
                    "ivs": {
                        stat: str(value)
                        for stat, value in zip(CANONICAL_STAT_KEYS, record.ivs)
                    },
                }
            )
        return projection

    def __repr__(self) -> str:
        return (
            "CanonicalTeamArtifact(team_id={!r}, format_id={!r}, "
            "artifact_schema_version={!r}, metadata_schema_version={!r}, "
            "sidecar_schema_version={!r}, set_count={!r})"
        ).format(
            self.team_id,
            self.format_id,
            self.artifact_schema_version,
            self.metadata_schema_version,
            self.sidecar_schema_version,
            self.set_count,
        )

    def __str__(self) -> str:
        return repr(self)


def _require_opaque_team_id(
    value: object,
    *,
    code: str,
    team_id: str | None = None,
) -> str:
    if not is_valid_opaque_team_id(value):
        _fail(code, "Canonical team ID is invalid", team_id=team_id)
    return value
