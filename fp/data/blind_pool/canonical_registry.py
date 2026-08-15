"""Dormant strict loading for externally bound canonical team registries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from .canonical_models import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_FORMAT_ID,
    CANONICAL_METADATA_SCHEMA_VERSION,
    CANONICAL_PROVISIONER_PROFILE_VERSION,
    CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION,
    CANONICAL_REGISTRY_SCHEMA_VERSION,
    CanonicalArtifactError,
    CanonicalArtifactMetadata,
    CanonicalRegistryEntry,
    CanonicalRuntimeRegistry,
    _CANONICAL_CONSTRUCTION_TOKEN,
    _fail,
    _require_exact_fields,
    _require_opaque_team_id,
    _strict_json_bytes,
)


_REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "registry_version",
        "format_id",
        "artifact_schema_version",
        "metadata_schema_version",
        "entries",
    }
)
_REGISTRY_ENTRY_FIELDS = frozenset({"team_id", "active", "metadata_sha256"})
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "team_id",
        "format_id",
        "format_fingerprint_sha256",
        "validation_status",
        "team_size",
        "showdown_commit",
        "showdown_tree_clean",
        "showdown_package_version",
        "package_lock_sha256",
        "dist_tree_sha256",
        "node_version",
        "npm_version",
        "provisioner_version",
        "provisioner_sha256",
        "source_sha256",
        "packed_sha256",
        "sidecar_sha256",
        "semantic_team_fingerprint_sha256",
    }
)
_ARTIFACT_NAMES = frozenset({"metadata.json", "packed.txt", "team.json"})
_REGISTRY_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)*$")
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")
_PACKAGE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}$")
_SEMVER_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_NODE_VERSION_PATTERN = re.compile(r"^v" + _SEMVER_PATTERN.pattern[1:])
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _is_within(path: Path, directory: Path, *, allow_equal: bool = True) -> bool:
    path_key = _path_key(path)
    directory_key = _path_key(directory)
    try:
        common = os.path.commonpath((path_key, directory_key))
    except ValueError:
        return False
    return common == directory_key and (allow_equal or path_key != directory_key)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        int(getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000))),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _same_file_object(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare stable identity without relying on settling timestamp fields."""

    return (
        int(left.st_dev),
        int(left.st_ino),
        stat.S_IFMT(left.st_mode),
        int(getattr(left, "st_file_attributes", 0)),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        stat.S_IFMT(right.st_mode),
        int(getattr(right, "st_file_attributes", 0)),
    )


def _resolve_existing_directory(
    path: Path,
    *,
    code: str,
    parent: Path | None = None,
    team_id: str | None = None,
) -> Path:
    if parent is not None and not _is_within(path, parent, allow_equal=False):
        _fail(code, "Canonical artifact path is invalid", team_id=team_id)
    try:
        before = os.lstat(path)
        if not stat.S_ISDIR(before.st_mode) or _is_link_or_reparse(before):
            _fail(code, "Canonical directory is invalid", team_id=team_id)
        resolved = path.resolve(strict=True)
        after = os.lstat(path)
    except CanonicalArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError):
        unavailable = True
    else:
        unavailable = False
    if unavailable:
        _fail(code, "Canonical directory is unavailable", team_id=team_id)
    if _stat_identity(before) != _stat_identity(after):
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical directory changed during verification",
            team_id=team_id,
        )
    if parent is not None and not _is_within(resolved, parent, allow_equal=False):
        _fail(code, "Canonical artifact path is invalid", team_id=team_id)
    return resolved


def _resolve_private_root(
    private_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    try:
        candidate = Path(private_root)
    except TypeError:
        candidate_invalid = True
    else:
        candidate_invalid = False
    if candidate_invalid:
        _fail("CANONICAL_ROOT_INVALID", "Canonical private root is invalid")
    if not candidate.is_absolute():
        _fail("CANONICAL_ROOT_INVALID", "Canonical private root must be absolute")
    resolved = _resolve_existing_directory(candidate, code="CANONICAL_ROOT_INVALID")
    try:
        repository = Path(repository_root or _repository_root()).resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        repository_unavailable = True
    else:
        repository_unavailable = False
    if repository_unavailable:
        _fail("CANONICAL_ROOT_INVALID", "Repository boundary is unavailable")
    if _is_within(resolved, repository) or _is_within(repository, resolved):
        _fail(
            "CANONICAL_ROOT_INVALID",
            "Canonical private root must remain outside the repository",
        )
    return resolved


def _resolve_registry_path(
    registry_path: str | Path,
    *,
    private_root: Path,
    repository_root: str | Path | None = None,
) -> Path:
    try:
        candidate = Path(registry_path)
    except TypeError:
        candidate_invalid = True
    else:
        candidate_invalid = False
    if candidate_invalid:
        _fail("CANONICAL_REGISTRY_INVALID", "Canonical registry path is invalid")
    if not candidate.is_absolute():
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry path must be absolute",
        )
    try:
        before = os.lstat(candidate)
        if not stat.S_ISREG(before.st_mode) or _is_link_or_reparse(before):
            _fail(
                "CANONICAL_REGISTRY_INVALID",
                "Canonical registry must be a regular file",
            )
        resolved = candidate.resolve(strict=True)
        after = os.lstat(candidate)
    except CanonicalArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError):
        unavailable = True
    else:
        unavailable = False
    if unavailable:
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry is unavailable",
        )
    if _stat_identity(before) != _stat_identity(after):
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical registry changed during verification",
        )
    try:
        repository = Path(repository_root or _repository_root()).resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        repository_unavailable = True
    else:
        repository_unavailable = False
    if repository_unavailable:
        _fail("CANONICAL_REGISTRY_INVALID", "Repository boundary is unavailable")
    if _is_within(resolved, repository):
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry must remain outside the repository",
        )
    canonical_root = private_root / "canonical"
    if _is_within(resolved, canonical_root):
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry must remain outside artifact directories",
        )
    return resolved


def _stable_read_bytes(
    path: Path,
    *,
    code: str,
    team_id: str | None = None,
) -> bytes:
    """Read one regular non-link file into an owned stable byte snapshot."""

    try:
        path_before = os.lstat(path)
        if not stat.S_ISREG(path_before.st_mode) or _is_link_or_reparse(path_before):
            _fail(code, "Canonical file is invalid", team_id=team_id)
        with path.open("rb") as source:
            descriptor_before = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(descriptor_before.st_mode)
                or _is_link_or_reparse(descriptor_before)
                or not _same_file_object(path_before, descriptor_before)
            ):
                _fail(code, "Canonical file is invalid", team_id=team_id)
            raw = source.read()
            descriptor_after = os.fstat(source.fileno())
        path_after = os.lstat(path)
    except CanonicalArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError):
        unreadable = True
    else:
        unreadable = False
    if unreadable:
        _fail(code, "Canonical file could not be read", team_id=team_id)
    if (
        _stat_identity(descriptor_before) != _stat_identity(descriptor_after)
        or _stat_identity(path_before) != _stat_identity(path_after)
        or not _same_file_object(descriptor_after, path_after)
        or len(raw) != descriptor_after.st_size
        or _is_link_or_reparse(path_after)
    ):
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical file changed during verification",
            team_id=team_id,
        )
    return bytes(raw)


@dataclass(frozen=True, slots=True)
class _DirectorySnapshot:
    directory_identity: tuple[int, ...]
    entries: tuple[tuple[str, tuple[int, ...]], ...]


def _artifact_directory_snapshot(
    directory: Path,
    *,
    team_id: str,
) -> _DirectorySnapshot:
    try:
        directory_before = os.lstat(directory)
        if not stat.S_ISDIR(directory_before.st_mode) or _is_link_or_reparse(
            directory_before
        ):
            _fail(
                "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                "Canonical artifact directory is invalid",
                team_id=team_id,
            )
        with os.scandir(directory) as iterator:
            scanned = list(iterator)
        names = {entry.name for entry in scanned}
        if names != _ARTIFACT_NAMES or len(scanned) != len(_ARTIFACT_NAMES):
            _fail(
                "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                "Canonical artifact directory contents are invalid",
                team_id=team_id,
            )
        entries: list[tuple[str, tuple[int, ...]]] = []
        for entry in scanned:
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or _is_link_or_reparse(info):
                _fail(
                    "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                    "Canonical artifact entry is invalid",
                    team_id=team_id,
                )
            entries.append((entry.name, _stat_identity(info)))
        directory_after = os.lstat(directory)
    except CanonicalArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError):
        unavailable = True
    else:
        unavailable = False
    if unavailable:
        _fail(
            "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
            "Canonical artifact directory could not be inspected",
            team_id=team_id,
        )
    if _stat_identity(directory_before) != _stat_identity(directory_after):
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical artifact directory changed during verification",
            team_id=team_id,
        )
    return _DirectorySnapshot(
        directory_identity=_stat_identity(directory_after),
        entries=tuple(sorted(entries)),
    )


def _resolve_schema_root(private_root: Path, artifact_schema_version: int) -> Path:
    canonical_root = _resolve_existing_directory(
        private_root / "canonical",
        code="CANONICAL_ARTIFACT_PATH_INVALID",
        parent=private_root,
    )
    return _resolve_existing_directory(
        canonical_root / "schema-{}".format(artifact_schema_version),
        code="CANONICAL_ARTIFACT_PATH_INVALID",
        parent=canonical_root,
    )


def _resolve_artifact_directory(
    private_root: Path,
    artifact_schema_version: int,
    team_id: str,
) -> Path:
    validated_team_id = _require_opaque_team_id(
        team_id,
        code="CANONICAL_ARTIFACT_PATH_INVALID",
    )
    schema_root = _resolve_schema_root(private_root, artifact_schema_version)
    return _resolve_existing_directory(
        schema_root / validated_team_id,
        code="CANONICAL_ARTIFACT_PATH_INVALID",
        parent=schema_root,
        team_id=validated_team_id,
    )


def _require_digest(value: Any, *, team_id: str | None = None) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        _fail(
            "CANONICAL_METADATA_INVALID",
            "Canonical metadata digest is invalid",
            team_id=team_id,
        )
    return value


def _parse_metadata(
    raw: bytes,
    *,
    expected_team_id: str,
    expected_schema_version: int,
) -> CanonicalArtifactMetadata:
    document = _strict_json_bytes(
        raw,
        encoding_code="CANONICAL_METADATA_INVALID",
        duplicate_code="CANONICAL_METADATA_INVALID",
        json_code="CANONICAL_METADATA_INVALID",
        team_id=expected_team_id,
    )
    value = _require_exact_fields(
        document,
        _METADATA_FIELDS,
        code="CANONICAL_METADATA_INVALID",
        context="Canonical metadata",
        team_id=expected_team_id,
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != (
        expected_schema_version
    ):
        _fail(
            "CANONICAL_METADATA_INVALID",
            "Canonical metadata schema is unsupported",
            team_id=expected_team_id,
        )
    team_id = _require_opaque_team_id(
        value["team_id"],
        code="CANONICAL_METADATA_INVALID",
        team_id=expected_team_id,
    )
    if team_id != expected_team_id:
        _fail(
            "CANONICAL_METADATA_INVALID",
            "Canonical metadata team binding is invalid",
            team_id=expected_team_id,
        )
    if (
        value["format_id"] != CANONICAL_FORMAT_ID
        or value["validation_status"] != "validated"
        or type(value["team_size"]) is not int
        or value["team_size"] != 6
        or value["showdown_tree_clean"] is not True
        or type(value["provisioner_version"]) is not int
        or value["provisioner_version"] != CANONICAL_PROVISIONER_PROFILE_VERSION
    ):
        _fail(
            "CANONICAL_METADATA_INVALID",
            "Canonical metadata policy is invalid",
            team_id=expected_team_id,
        )
    if (
        not isinstance(value["showdown_commit"], str)
        or _GIT_COMMIT_PATTERN.fullmatch(value["showdown_commit"]) is None
        or not isinstance(value["showdown_package_version"], str)
        or _PACKAGE_VERSION_PATTERN.fullmatch(value["showdown_package_version"]) is None
        or not isinstance(value["node_version"], str)
        or _NODE_VERSION_PATTERN.fullmatch(value["node_version"]) is None
        or not isinstance(value["npm_version"], str)
        or _SEMVER_PATTERN.fullmatch(value["npm_version"]) is None
    ):
        _fail(
            "CANONICAL_METADATA_INVALID",
            "Canonical metadata provenance is invalid",
            team_id=expected_team_id,
        )
    digest_fields = (
        "format_fingerprint_sha256",
        "package_lock_sha256",
        "dist_tree_sha256",
        "provisioner_sha256",
        "source_sha256",
        "packed_sha256",
        "sidecar_sha256",
        "semantic_team_fingerprint_sha256",
    )
    digests = {
        field_name: _require_digest(value[field_name], team_id=expected_team_id)
        for field_name in digest_fields
    }
    return CanonicalArtifactMetadata(
        schema_version=value["schema_version"],
        team_id=team_id,
        format_id=CANONICAL_FORMAT_ID,
        format_fingerprint_sha256=digests["format_fingerprint_sha256"],
        validation_status="validated",
        team_size=6,
        showdown_commit=value["showdown_commit"],
        showdown_tree_clean=True,
        showdown_package_version=value["showdown_package_version"],
        package_lock_sha256=digests["package_lock_sha256"],
        dist_tree_sha256=digests["dist_tree_sha256"],
        node_version=value["node_version"],
        npm_version=value["npm_version"],
        provisioner_version=CANONICAL_PROVISIONER_PROFILE_VERSION,
        provisioner_sha256=digests["provisioner_sha256"],
        source_sha256=digests["source_sha256"],
        packed_sha256=digests["packed_sha256"],
        sidecar_sha256=digests["sidecar_sha256"],
        semantic_team_fingerprint_sha256=digests["semantic_team_fingerprint_sha256"],
    )


def _load_bound_metadata(
    private_root: Path,
    artifact_schema_version: int,
    metadata_schema_version: int,
    entry: CanonicalRegistryEntry,
) -> tuple[Path, CanonicalArtifactMetadata, _DirectorySnapshot]:
    directory = _resolve_artifact_directory(
        private_root,
        artifact_schema_version,
        entry.team_id,
    )
    initial_snapshot = _artifact_directory_snapshot(directory, team_id=entry.team_id)
    raw = _stable_read_bytes(
        directory / "metadata.json",
        code="CANONICAL_METADATA_INVALID",
        team_id=entry.team_id,
    )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_digest, entry.metadata_sha256):
        _fail(
            "CANONICAL_METADATA_INTEGRITY_MISMATCH",
            "Canonical metadata failed its integrity check",
            team_id=entry.team_id,
        )
    metadata = _parse_metadata(
        raw,
        expected_team_id=entry.team_id,
        expected_schema_version=metadata_schema_version,
    )
    final_snapshot = _artifact_directory_snapshot(directory, team_id=entry.team_id)
    if final_snapshot != initial_snapshot:
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical artifact directory changed during verification",
            team_id=entry.team_id,
        )
    return directory, metadata, final_snapshot


def _parse_registry_document(
    document: Any,
) -> tuple[
    int,
    str,
    str,
    int,
    int,
    tuple[CanonicalRegistryEntry, ...],
]:
    value = _require_exact_fields(
        document,
        _REGISTRY_FIELDS,
        code="CANONICAL_REGISTRY_INVALID",
        context="Canonical runtime registry",
    )
    schema_version = value["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != CANONICAL_REGISTRY_SCHEMA_VERSION
    ):
        _fail(
            "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED",
            "Canonical runtime registry schema is unsupported",
        )
    registry_version = value["registry_version"]
    if (
        not isinstance(registry_version, str)
        or _REGISTRY_VERSION_PATTERN.fullmatch(registry_version) is None
    ):
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry deployment version is invalid",
        )
    if value["format_id"] != CANONICAL_FORMAT_ID:
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry format is incompatible",
        )
    if (
        type(value["artifact_schema_version"]) is not int
        or value["artifact_schema_version"] != CANONICAL_ARTIFACT_SCHEMA_VERSION
        or type(value["metadata_schema_version"]) is not int
        or value["metadata_schema_version"] != CANONICAL_METADATA_SCHEMA_VERSION
    ):
        _fail(
            "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED",
            "Canonical artifact or metadata schema is unsupported",
        )
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry entries must be a nonempty array",
        )
    entries: list[CanonicalRegistryEntry] = []
    seen_ids: set[str] = set()
    for raw_entry in entries_value:
        entry_value = _require_exact_fields(
            raw_entry,
            _REGISTRY_ENTRY_FIELDS,
            code="CANONICAL_REGISTRY_INVALID",
            context="Canonical registry entry",
        )
        team_id = _require_opaque_team_id(
            entry_value["team_id"],
            code="CANONICAL_REGISTRY_INVALID",
        )
        if team_id in seen_ids:
            _fail(
                "CANONICAL_REGISTRY_INVALID",
                "Canonical registry contains duplicate team IDs",
                team_id=team_id,
            )
        seen_ids.add(team_id)
        active = entry_value["active"]
        if type(active) is not bool:
            _fail(
                "CANONICAL_REGISTRY_INVALID",
                "Canonical registry active flag is invalid",
                team_id=team_id,
            )
        metadata_sha256 = entry_value["metadata_sha256"]
        if (
            not isinstance(metadata_sha256, str)
            or _DIGEST_PATTERN.fullmatch(metadata_sha256) is None
        ):
            _fail(
                "CANONICAL_REGISTRY_INVALID",
                "Canonical registry metadata digest is invalid",
                team_id=team_id,
            )
        entries.append(CanonicalRegistryEntry(team_id, active, metadata_sha256))
    if not any(entry.active for entry in entries):
        _fail(
            "CANONICAL_REGISTRY_INVALID",
            "Canonical registry must contain an active entry",
        )
    return (
        schema_version,
        registry_version,
        CANONICAL_FORMAT_ID,
        CANONICAL_ARTIFACT_SCHEMA_VERSION,
        CANONICAL_METADATA_SCHEMA_VERSION,
        tuple(sorted(entries, key=lambda item: item.team_id)),
    )


def _fingerprint_projection(
    *,
    schema_version: int,
    registry_version: str,
    format_id: str,
    artifact_schema_version: int,
    metadata_schema_version: int,
    entries: tuple[CanonicalRegistryEntry, ...],
) -> dict[str, Any]:
    return {
        "domain": "foul-play-tugs-canonical-registry",
        "fingerprint_profile_version": (CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION),
        "registry_schema_version": schema_version,
        "registry_version": registry_version,
        "format_id": format_id,
        "artifact_schema_version": artifact_schema_version,
        "metadata_schema_version": metadata_schema_version,
        "entries": [
            {
                "team_id": entry.team_id,
                "active": entry.active,
                "metadata_sha256": entry.metadata_sha256,
            }
            for entry in sorted(entries, key=lambda item: item.team_id)
        ],
    }


def _compute_fingerprint(
    *,
    schema_version: int,
    registry_version: str,
    format_id: str,
    artifact_schema_version: int,
    metadata_schema_version: int,
    entries: tuple[CanonicalRegistryEntry, ...],
) -> str:
    try:
        canonical = json.dumps(
            _fingerprint_projection(
                schema_version=schema_version,
                registry_version=registry_version,
                format_id=format_id,
                artifact_schema_version=artifact_schema_version,
                metadata_schema_version=metadata_schema_version,
                entries=entries,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        failed = True
    else:
        failed = False
    if failed:
        _fail(
            "CANONICAL_REGISTRY_FINGERPRINT_ERROR",
            "Canonical registry fingerprint could not be computed",
        )
    return hashlib.sha256(canonical).hexdigest()


def compute_canonical_registry_fingerprint(
    registry: CanonicalRuntimeRegistry,
) -> str:
    """Compute the path-independent semantic fingerprint for one registry."""

    if not isinstance(registry, CanonicalRuntimeRegistry):
        _fail(
            "CANONICAL_REGISTRY_FINGERPRINT_ERROR",
            "Canonical registry has an invalid type",
        )
    return _compute_fingerprint(
        schema_version=registry.schema_version,
        registry_version=registry.registry_version,
        format_id=registry.format_id,
        artifact_schema_version=registry.artifact_schema_version,
        metadata_schema_version=registry.metadata_schema_version,
        entries=registry.entries,
    )


def load_canonical_runtime_registry(
    private_root: str | Path,
    registry_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> CanonicalRuntimeRegistry:
    """Explicitly load a dormant canonical registry and bind every metadata file."""

    resolved_root = _resolve_private_root(
        private_root,
        repository_root=repository_root,
    )
    resolved_registry = _resolve_registry_path(
        registry_path,
        private_root=resolved_root,
        repository_root=repository_root,
    )
    raw = _stable_read_bytes(
        resolved_registry,
        code="CANONICAL_REGISTRY_INVALID",
    )
    document = _strict_json_bytes(
        raw,
        encoding_code="CANONICAL_REGISTRY_INVALID",
        duplicate_code="CANONICAL_REGISTRY_DUPLICATE_KEY",
        json_code="CANONICAL_REGISTRY_INVALID",
    )
    (
        schema_version,
        registry_version,
        format_id,
        artifact_schema_version,
        metadata_schema_version,
        entries,
    ) = _parse_registry_document(document)

    metadata_items: list[CanonicalArtifactMetadata] = []
    active_provenance: tuple[object, ...] | None = None
    for entry in entries:
        _, metadata, _ = _load_bound_metadata(
            resolved_root,
            artifact_schema_version,
            metadata_schema_version,
            entry,
        )
        metadata_items.append(metadata)
        if entry.active:
            if active_provenance is None:
                active_provenance = metadata.active_provenance
            elif metadata.active_provenance != active_provenance:
                _fail(
                    "CANONICAL_ACTIVE_PROVENANCE_MISMATCH",
                    "Active canonical artifacts use inconsistent provenance",
                    team_id=entry.team_id,
                )

    fingerprint = _compute_fingerprint(
        schema_version=schema_version,
        registry_version=registry_version,
        format_id=format_id,
        artifact_schema_version=artifact_schema_version,
        metadata_schema_version=metadata_schema_version,
        entries=entries,
    )
    return CanonicalRuntimeRegistry(
        schema_version=schema_version,
        registry_version=registry_version,
        format_id=format_id,
        artifact_schema_version=artifact_schema_version,
        metadata_schema_version=metadata_schema_version,
        entries=entries,
        _private_root=resolved_root,
        _metadata=tuple(metadata_items),
        _fingerprint=fingerprint,
        _construction_token=_CANONICAL_CONSTRUCTION_TOKEN,
    )
