"""Selected-team verification for dormant canonical Blind Ladder artifacts."""

from __future__ import annotations

import hashlib
import hmac

from .canonical_models import (
    CanonicalRuntimeRegistry,
    CanonicalTeamArtifact,
    _CANONICAL_CONSTRUCTION_TOKEN,
    _fail,
    _require_opaque_team_id,
)
from .canonical_registry import (
    _artifact_directory_snapshot,
    _load_bound_metadata,
    _stable_read_bytes,
)
from .canonical_sidecar import parse_canonical_sidecar


def _verification_boundary(_name: str, _team_id: str) -> None:
    """Private no-op seam used to prove fail-closed concurrent-change checks."""


def _decode_packed(raw: bytes, *, team_id: str) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(
            "CANONICAL_PACKED_INVALID",
            "Canonical packed bytes are invalid",
            team_id=team_id,
        )
    try:
        packed = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        packed = None
    if packed is None:
        _fail(
            "CANONICAL_PACKED_INVALID",
            "Canonical packed bytes are invalid",
            team_id=team_id,
        )
    if (
        not packed
        or "\x00" in packed
        or "\r" in packed
        or "\n" in packed
        or "\ufeff" in packed
    ):
        _fail(
            "CANONICAL_PACKED_INVALID",
            "Canonical packed value is invalid",
            team_id=team_id,
        )
    return packed


def load_canonical_team_artifact(
    registry: CanonicalRuntimeRegistry,
    team_id: str,
) -> CanonicalTeamArtifact:
    """Verify and adapt exactly one registry-referenced canonical artifact."""

    if not isinstance(registry, CanonicalRuntimeRegistry):
        _fail(
            "CANONICAL_INTERNAL_ERROR",
            "Canonical registry has an invalid type",
        )
    selected_team_id = _require_opaque_team_id(
        team_id,
        code="CANONICAL_ARTIFACT_PATH_INVALID",
    )
    entry = registry.get_entry(selected_team_id)
    trusted_metadata = registry._metadata_for(selected_team_id)
    if entry is None or trusted_metadata is None:
        _fail(
            "CANONICAL_ARTIFACT_PATH_INVALID",
            "Canonical team is not registered",
            team_id=selected_team_id,
        )

    directory, metadata, initial_snapshot = _load_bound_metadata(
        registry._private_root,
        registry.artifact_schema_version,
        registry.metadata_schema_version,
        entry,
    )
    if metadata != trusted_metadata:
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical metadata changed after registry loading",
            team_id=selected_team_id,
        )
    _verification_boundary("metadata_verified", selected_team_id)

    packed_raw = _stable_read_bytes(
        directory / "packed.txt",
        code="CANONICAL_PACKED_INVALID",
        team_id=selected_team_id,
    )
    packed_digest = hashlib.sha256(packed_raw).hexdigest()
    if not hmac.compare_digest(packed_digest, metadata.packed_sha256):
        _fail(
            "CANONICAL_PACKED_INTEGRITY_MISMATCH",
            "Canonical packed bytes failed their integrity check",
            team_id=selected_team_id,
        )
    packed = _decode_packed(packed_raw, team_id=selected_team_id)
    _verification_boundary("packed_verified", selected_team_id)

    sidecar_raw = _stable_read_bytes(
        directory / "team.json",
        code="CANONICAL_SIDECAR_INVALID",
        team_id=selected_team_id,
    )
    sidecar_digest = hashlib.sha256(sidecar_raw).hexdigest()
    if not hmac.compare_digest(sidecar_digest, metadata.sidecar_sha256):
        _fail(
            "CANONICAL_SIDECAR_INTEGRITY_MISMATCH",
            "Canonical sidecar failed its integrity check",
            team_id=selected_team_id,
        )
    sidecar = parse_canonical_sidecar(
        sidecar_raw,
        expected_team_id=selected_team_id,
    )
    _verification_boundary("sidecar_verified", selected_team_id)

    final_snapshot = _artifact_directory_snapshot(
        directory,
        team_id=selected_team_id,
    )
    if final_snapshot != initial_snapshot:
        _fail(
            "CANONICAL_ARTIFACT_CHANGED",
            "Canonical artifact changed during verification",
            team_id=selected_team_id,
        )
    return CanonicalTeamArtifact(
        team_id=selected_team_id,
        format_id=registry.format_id,
        artifact_schema_version=registry.artifact_schema_version,
        metadata_schema_version=registry.metadata_schema_version,
        sidecar_schema_version=sidecar.schema_version,
        packed_wire=packed,
        records=sidecar.records,
        construction_token=_CANONICAL_CONSTRUCTION_TOKEN,
    )
