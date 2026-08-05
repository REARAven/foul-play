"""Immutable schema-v1 public TUGS opponent priors with no runtime integration.

Accepted documents contain top-level ``schema_version``, ``visibility``,
``dataset_id``, ``dataset_version``, ``format_id``, ``sources``, and ``species``
fields, plus optional patch/display/metadata fields.  Visibility must be exactly
``"public"``.  Each source has an explicitly public kind.  Each exact species or
form owns one or more complete four-move set variants whose authored positive
weights can be normalized through a read-only derived API.

The declaration cannot prove real-world public visibility.  The caller or an
offline import process remains responsible for source authorization and for
loading the correct format overlay before validation.  Item existence and move
learnset legality are not guessed because this repository lacks authoritative
complete registries for those questions.
"""

from .errors import (
    PublicPriorError,
    PublicPriorRegistryError,
    PublicPriorValidationError,
    ValidationIssue,
)
from .loader import load_public_prior
from .models import (
    NO_ITEM_ID,
    PUBLIC_VISIBILITY,
    SCHEMA_VERSION,
    NormalizedVariantProbability,
    PublicPriorDataset,
    PublicPriorIdentity,
    PublicPriorRegistry,
    PublicSetVariant,
    PublicSource,
    PublicSourceKind,
    PublicStatValues,
    PublicVariantReference,
    SpeciesPrior,
)
from .validation import validate_public_prior_document

__all__ = (
    "NO_ITEM_ID",
    "PUBLIC_VISIBILITY",
    "SCHEMA_VERSION",
    "NormalizedVariantProbability",
    "PublicPriorDataset",
    "PublicPriorError",
    "PublicPriorIdentity",
    "PublicPriorRegistry",
    "PublicPriorRegistryError",
    "PublicPriorValidationError",
    "PublicSetVariant",
    "PublicSource",
    "PublicSourceKind",
    "PublicStatValues",
    "PublicVariantReference",
    "SpeciesPrior",
    "ValidationIssue",
    "load_public_prior",
    "validate_public_prior_document",
)
