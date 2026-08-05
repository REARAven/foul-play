"""Side-effect-free JSON file loading for immutable team pools.

Only the explicitly supplied UTF-8 file is read.  This module performs no
overlay switching, network access, caching, source rewriting, or battle
integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import TeamPoolValidationError, ValidationIssue
from .models import TeamPool
from .validation import JsonObjectPairs, validate_team_pool_document


def _pairs_hook(pairs: list[tuple[str, Any]]) -> JsonObjectPairs:
    return JsonObjectPairs(pairs)


def load_team_pool(path: str | Path) -> TeamPool:
    """Load and validate one schema-version-1 team-pool JSON file."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        raise TeamPoolValidationError(
            (ValidationIssue("$", f"team-pool file does not exist: {source}"),)
        ) from None
    except UnicodeDecodeError as error:
        raise TeamPoolValidationError(
            (
                ValidationIssue(
                    "$", f"team-pool file is not valid UTF-8 at byte {error.start}"
                ),
            )
        ) from None
    except OSError as error:
        raise TeamPoolValidationError(
            (ValidationIssue("$", f"could not read team-pool file: {error}"),)
        ) from None

    try:
        document = json.loads(raw, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as error:
        raise TeamPoolValidationError(
            (
                ValidationIssue(
                    "$",
                    f"malformed JSON at line {error.lineno}, column {error.colno}: {error.msg}",
                ),
            )
        ) from None

    return validate_team_pool_document(document, source_path=str(source))
