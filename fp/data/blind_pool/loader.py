"""Strict, side-effect-free loading of an external Blind Ladder registry.

Registry entries use opaque IDs so no content-derived label is needed. Team
paths are resolved beneath the configured private root before hashing to stop
relative paths and symlinks from escaping that boundary. SHA-256 covers the
exact source-file bytes; this phase verifies files but intentionally does not
parse, select, reserve, or provision teams.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from .config import _is_within, validate_blind_pool_config
from .errors import BlindPoolValidationError
from .models import (
    OPAQUE_TEAM_ID_PATTERN,
    SCHEMA_VERSION,
    SUPPORTED_FORMAT_ID,
    BlindPoolConfig,
    BlindPoolEntry,
    BlindPoolRegistry,
)


logger = logging.getLogger(__name__)

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "registry_version", "format_id", "entries"}
)
_ENTRY_FIELDS = frozenset({"team_id", "active", "team_file", "sha256"})
_TEAM_ID_PATTERN = OPAQUE_TEAM_ID_PATTERN
_REGISTRY_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_PATH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


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


def _fail(
    code: str,
    message: str,
    *,
    team_id: str | None = None,
) -> NoReturn:
    raise BlindPoolValidationError(code, message, team_id=team_id) from None


def _validate_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    context: str,
    team_id: str | None = None,
) -> None:
    missing = expected - value.keys()
    if missing:
        _fail(
            "missing_required_field",
            "{} is missing required fields".format(context),
            team_id=team_id,
        )
    unexpected = value.keys() - expected
    if unexpected:
        _fail(
            "unexpected_field",
            "{} contains unexpected fields".format(context),
            team_id=team_id,
        )


def _validate_team_id(value: Any) -> str:
    if not isinstance(value, str) or _TEAM_ID_PATTERN.fullmatch(value) is None:
        _fail(
            "team_id_invalid",
            "Blind Ladder team ID must be an opaque ID such as BL-001-v1",
        )
    return value


def _validate_relative_team_path(value: Any, team_id: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(
            "team_path_invalid",
            "Team file path must be a nonempty portable relative path",
            team_id=team_id,
        )
    raw_parts = value.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        _fail(
            "team_path_traversal",
            "Team file path must remain relative to the private root",
            team_id=team_id,
        )
    path = PurePosixPath(value)
    if path.is_absolute():
        _fail(
            "team_path_traversal",
            "Team file path must remain relative to the private root",
            team_id=team_id,
        )
    if any(
        _PORTABLE_PATH_SEGMENT_PATTERN.fullmatch(part) is None
        or part.endswith((" ", "."))
        for part in path.parts
    ):
        _fail(
            "team_path_invalid",
            "Team file path contains an unsupported path segment",
            team_id=team_id,
        )
    return path


def _resolve_team_path(
    private_root: Path,
    relative_path: PurePosixPath,
    team_id: str,
) -> Path:
    candidate = private_root.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail(
            "team_file_missing",
            "Referenced team file is unavailable",
            team_id=team_id,
        )
    if not _is_within(resolved, private_root):
        _fail(
            "team_path_escape",
            "Resolved team file escapes the configured private root",
            team_id=team_id,
        )
    if not resolved.is_file():
        _fail(
            "team_file_invalid",
            "Referenced team file must be a regular file",
            team_id=team_id,
        )
    return resolved


def _sha256_file(path: Path, team_id: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        _fail(
            "team_file_unreadable",
            "Referenced team file could not be read",
            team_id=team_id,
        )
    return digest.hexdigest()


def _validate_entry(
    value: Any,
    *,
    private_root: Path,
    index: int,
) -> BlindPoolEntry:
    if not isinstance(value, dict):
        _fail(
            "entry_type_invalid",
            "Registry entry {} must be an object".format(index),
        )
    candidate_team_id = value.get("team_id")
    team_id = (
        candidate_team_id
        if isinstance(candidate_team_id, str)
        and _TEAM_ID_PATTERN.fullmatch(candidate_team_id) is not None
        else None
    )
    _validate_exact_fields(
        value,
        _ENTRY_FIELDS,
        context="Registry entry {}".format(index),
        team_id=team_id,
    )
    team_id = _validate_team_id(value["team_id"])
    active = value["active"]
    if not isinstance(active, bool):
        _fail(
            "active_flag_invalid",
            "Entry active flag must be a boolean",
            team_id=team_id,
        )
    relative_path = _validate_relative_team_path(value["team_file"], team_id)
    expected_sha256 = value["sha256"]
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        _fail(
            "sha256_invalid",
            "Entry SHA-256 must contain 64 lowercase hexadecimal characters",
            team_id=team_id,
        )

    resolved_path = _resolve_team_path(private_root, relative_path, team_id)
    actual_sha256 = _sha256_file(resolved_path, team_id)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        _fail(
            "sha256_mismatch",
            "Referenced team file failed its integrity check",
            team_id=team_id,
        )

    return BlindPoolEntry(
        team_id=team_id,
        active=active,
        relative_team_path=str(relative_path),
        resolved_team_path=resolved_path,
        sha256=expected_sha256,
    )


def validate_blind_pool_registry(
    document: Any,
    config: BlindPoolConfig,
) -> BlindPoolRegistry:
    """Validate one decoded schema-v1 registry and every referenced team file."""

    validated_config = validate_blind_pool_config(config)
    if not isinstance(document, dict):
        _fail("registry_type_invalid", "Blind Ladder registry must be an object")
    _validate_exact_fields(
        document,
        _TOP_LEVEL_FIELDS,
        context="Blind Ladder registry",
    )

    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _fail(
            "schema_version_unsupported",
            "Blind Ladder registry schema version is unsupported",
        )
    registry_version = document["registry_version"]
    if (
        not isinstance(registry_version, str)
        or _REGISTRY_VERSION_PATTERN.fullmatch(registry_version) is None
    ):
        _fail(
            "registry_version_invalid",
            "Blind Ladder registry version must be a dotted numeric string",
        )
    if document["format_id"] != SUPPORTED_FORMAT_ID:
        _fail(
            "format_incompatible",
            "Blind Ladder registry format must be gen9tugs",
        )
    entries_value = document["entries"]
    if not isinstance(entries_value, list):
        _fail("entries_type_invalid", "Blind Ladder entries must be an array")

    entries: list[BlindPoolEntry] = []
    seen_ids: set[str] = set()
    active_paths: set[Path] = set()
    for index, entry_value in enumerate(entries_value):
        entry = _validate_entry(
            entry_value,
            private_root=validated_config.private_root,
            index=index,
        )
        if entry.team_id in seen_ids:
            _fail(
                "duplicate_team_id",
                "Blind Ladder registry contains duplicate opaque team IDs",
                team_id=entry.team_id,
            )
        seen_ids.add(entry.team_id)
        if entry.active:
            if entry.resolved_team_path in active_paths:
                _fail(
                    "duplicate_active_team_file",
                    "Active entries must reference unique team files",
                    team_id=entry.team_id,
                )
            active_paths.add(entry.resolved_team_path)
        entries.append(entry)

    if not any(entry.active for entry in entries):
        _fail(
            "active_pool_empty",
            "Blind Ladder registry must contain at least one active entry",
        )

    return BlindPoolRegistry(
        schema_version=schema_version,
        registry_version=registry_version,
        format_id=SUPPORTED_FORMAT_ID,
        entries=tuple(entries),
    )


def load_blind_pool_registry(config: BlindPoolConfig) -> BlindPoolRegistry:
    """Read and validate exactly one configured external registry file."""

    validated_config = validate_blind_pool_config(config)
    try:
        raw = validated_config.registry_path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError:
        _fail(
            "registry_encoding_invalid",
            "Blind Ladder registry is not valid UTF-8",
        )
    except OSError:
        _fail(
            "registry_file_unreadable",
            "Blind Ladder registry could not be read",
        )

    try:
        document = json.loads(
            raw,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJsonFieldError:
        _fail(
            "registry_duplicate_field",
            "Blind Ladder registry JSON contains a duplicate field",
        )
    except (json.JSONDecodeError, ValueError):
        _fail(
            "registry_json_invalid",
            "Blind Ladder registry contains malformed JSON",
        )

    registry = validate_blind_pool_registry(document, validated_config)
    logger.info(
        "Loaded Blind Ladder registry schema={} version={} format={} "
        "entries={} active={}".format(
            registry.schema_version,
            registry.registry_version,
            registry.format_id,
            len(registry),
            len(registry.active_entries),
        )
    )
    return registry


def get_active_blind_pool_entries(
    registry: BlindPoolRegistry,
) -> tuple[BlindPoolEntry, ...]:
    """Return active entries in stable registry order without selecting one."""

    if not isinstance(registry, BlindPoolRegistry):
        _fail(
            "registry_type_invalid",
            "Blind Ladder registry has an invalid type",
        )
    return registry.active_entries


def get_blind_pool_entry_by_id(
    registry: BlindPoolRegistry,
    team_id: str,
) -> BlindPoolEntry | None:
    """Resolve one opaque ID without performing selection or reservation."""

    if not isinstance(registry, BlindPoolRegistry):
        _fail(
            "registry_type_invalid",
            "Blind Ladder registry has an invalid type",
        )
    if not isinstance(team_id, str) or _TEAM_ID_PATTERN.fullmatch(team_id) is None:
        _fail(
            "team_id_invalid",
            "Blind Ladder team ID must be an opaque ID such as BL-001-v1",
        )
    return registry.get_entry(team_id)
