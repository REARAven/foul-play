"""Side-effect-free local JSON loading for immutable public priors.

Only the explicitly supplied UTF-8 file is read.  There is no overlay switch,
private-pool lookup, network access, cache access, output file, or battle/search
integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import PublicPriorValidationError, ValidationIssue
from .models import PublicPriorDataset
from .validation import JsonObjectPairs, validate_public_prior_document


def _pairs_hook(pairs: list[tuple[str, Any]]) -> JsonObjectPairs:
    return JsonObjectPairs(pairs)


def load_public_prior(path: str | Path) -> PublicPriorDataset:
    """Read and validate exactly one schema-version-1 public-prior file."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        raise PublicPriorValidationError(
            (ValidationIssue("$", f"public-prior file does not exist: {source}"),)
        ) from None
    except UnicodeDecodeError as error:
        raise PublicPriorValidationError(
            (
                ValidationIssue(
                    "$", f"public-prior file is not valid UTF-8 at byte {error.start}"
                ),
            )
        ) from None
    except OSError as error:
        raise PublicPriorValidationError(
            (ValidationIssue("$", f"could not read public-prior file: {error}"),)
        ) from None

    try:
        document = json.loads(raw, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as error:
        raise PublicPriorValidationError(
            (
                ValidationIssue(
                    "$",
                    f"malformed JSON at line {error.lineno}, column {error.colno}: {error.msg}",
                ),
            )
        ) from None

    return validate_public_prior_document(document, source_path=str(source))
