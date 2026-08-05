"""Immutable schema-v1 complete-team pools with no battle integration.

The public API loads only explicit local JSON files, performs no network or
cache activity, preserves exact species forms in roster identity, and retains
all matching team variants.  Public-field metadata is stored without applying
an open- or closed-sheet policy.

Schema v1 accepts ``{"schema_version": 1, "pool": {...}, "teams": [...]}``.
The pool object requires ``pool_id``, ``pool_version``, and ``format_id`` and
may carry patch/source/display/public-field/metadata values.  Every team
requires stable team and variant IDs plus exactly six complete Pokémon; every
Pokémon requires stable slot/species/item/ability/move/nature/stat/level data.
The exact no-item sentinel is ``"none"``.  Metadata and public-field arrays are
optional, and a sorted source ``roster_key`` may be supplied for verification.
"""

from .errors import (
    TeamPoolError,
    TeamPoolRegistryError,
    TeamPoolValidationError,
    ValidationIssue,
)
from .loader import load_team_pool
from .models import (
    NO_ITEM_ID,
    PUBLIC_FIELD_NAMES,
    SCHEMA_VERSION,
    PokemonRecord,
    PoolIdentity,
    RosterKey,
    SourceLocation,
    StatValues,
    TeamPool,
    TeamPoolRegistry,
    TeamRecord,
    TeamRecordId,
    canonical_roster_key,
)
from .validation import validate_team_pool_document

__all__ = (
    "NO_ITEM_ID",
    "PUBLIC_FIELD_NAMES",
    "SCHEMA_VERSION",
    "PokemonRecord",
    "PoolIdentity",
    "RosterKey",
    "SourceLocation",
    "StatValues",
    "TeamPool",
    "TeamPoolError",
    "TeamPoolRegistry",
    "TeamPoolRegistryError",
    "TeamPoolValidationError",
    "TeamRecord",
    "TeamRecordId",
    "ValidationIssue",
    "canonical_roster_key",
    "load_team_pool",
    "validate_team_pool_document",
)
