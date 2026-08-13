"""Deterministic semantic identity for a validated Blind Ladder registry."""

from __future__ import annotations

import hashlib
import json

from .errors import BlindPoolValidationError
from .models import BlindPoolRegistry


def compute_registry_fingerprint(registry: BlindPoolRegistry) -> str:
    """Hash portable validated registry semantics, never filesystem locations."""

    if not isinstance(registry, BlindPoolRegistry):
        raise BlindPoolValidationError(
            "registry_type_invalid",
            "Blind Ladder registry has an invalid type",
        ) from None
    document = {
        "schema_version": registry.schema_version,
        "registry_version": registry.registry_version,
        "format_id": registry.format_id,
        "entries": [
            {
                "team_id": entry.team_id,
                "active": entry.active,
                "team_file": entry.relative_team_path,
                "sha256": entry.sha256,
            }
            for entry in sorted(registry.entries, key=lambda item: item.team_id)
        ],
    }
    canonical = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
