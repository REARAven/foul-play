"""Immutable, explicitly configured public-prior search context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fp.battle.helpers import normalize_name
from fp.data.public_priors import (
    PublicPriorIdentity,
    PublicPriorRegistry,
)


class PublicPriorFallback(Enum):
    """The only permitted behavior after a public-prior miss."""

    GENERIC = "generic"
    NONE = "none"


class SelectedDatasetStatus(Enum):
    """Safe construction-time status for one selected public identity."""

    AVAILABLE = "available"
    MISSING = "missing"
    FORMAT_MISMATCH = "format_mismatch"


@dataclass(frozen=True)
class SelectedDatasetDiagnostic:
    """Public-only diagnostic with no opponent or private-team identity."""

    identity: PublicPriorIdentity
    status: SelectedDatasetStatus
    detail: str


@dataclass(frozen=True)
class PublicPriorSearchContext:
    """Battle-local public-prior configuration safe to share across copies.

    ``fallback_policy`` has no default: constructing a configured context always
    requires the caller to choose GENERIC or NONE explicitly.
    """

    registry: PublicPriorRegistry
    selected_identities: tuple[PublicPriorIdentity, ...]
    fallback_policy: PublicPriorFallback
    format_id: str
    diagnostics: tuple[SelectedDatasetDiagnostic, ...] = field(
        init=False, repr=True
    )

    def __post_init__(self) -> None:
        if not isinstance(self.registry, PublicPriorRegistry):
            raise TypeError("registry must be a PublicPriorRegistry")
        identities = tuple(self.selected_identities)
        if not all(
            isinstance(identity, PublicPriorIdentity) for identity in identities
        ):
            raise TypeError("selected identities must be PublicPriorIdentity values")
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate selected public-prior identity")
        if not isinstance(self.fallback_policy, PublicPriorFallback):
            raise TypeError(
                "fallback_policy must be PublicPriorFallback.GENERIC or NONE"
            )
        if (
            not isinstance(self.format_id, str)
            or not self.format_id
            or normalize_name(self.format_id) != self.format_id
        ):
            raise ValueError("format_id must be a nonempty canonical normalized ID")

        diagnostics = []
        for identity in identities:
            if identity.format_id != self.format_id:
                status = SelectedDatasetStatus.FORMAT_MISMATCH
                detail = "selected identity format does not match context format"
            elif self.registry.get(identity) is None:
                status = SelectedDatasetStatus.MISSING
                detail = "selected identity is absent from the public registry"
            else:
                status = SelectedDatasetStatus.AVAILABLE
                detail = "selected public dataset is available and format-compatible"
            diagnostics.append(SelectedDatasetDiagnostic(identity, status, detail))

        object.__setattr__(self, "selected_identities", identities)
        object.__setattr__(self, "diagnostics", tuple(diagnostics))

    def __deepcopy__(self, memo: dict[int, Any]) -> PublicPriorSearchContext:
        """Share only this immutable context and its immutable public records."""

        memo[id(self)] = self
        return self
