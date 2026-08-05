"""Stable errors produced by the public-opponent-prior data layer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ValidationIssue:
    """One source-data problem with immutable, machine-readable context."""

    path: str
    explanation: str
    dataset_id: str | None = None
    species_id: str | None = None
    variant_id: str | None = None
    field: str | None = None
    invalid_value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "invalid_value", _freeze_value(self.invalid_value))

    def __str__(self) -> str:
        context = []
        for name in ("dataset_id", "species_id", "variant_id", "field"):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        if self.invalid_value is not None:
            context.append(f"invalid_value={self.invalid_value!r}")
        suffix = f" [{', '.join(context)}]" if context else ""
        return f"{self.path}: {self.explanation}{suffix}"


class PublicPriorError(Exception):
    """Base class for public-prior errors."""


class PublicPriorValidationError(PublicPriorError):
    """All independently detectable validation issues for one document."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("PublicPriorValidationError requires at least one issue")
        super().__init__("\n".join(str(issue) for issue in self.issues))


class PublicPriorRegistryError(PublicPriorError):
    """An immutable registry could not be built unambiguously."""
