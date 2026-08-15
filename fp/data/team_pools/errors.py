"""Stable errors produced by the team-pool data layer."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ValidationIssue:
    """One source-data problem with machine-readable location context."""

    path: str
    message: str
    pool_id: str | None = None
    team_id: str | None = None
    variant_id: str | None = None
    slot_id: str | None = None
    field_name: str | None = None
    invalid_value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "invalid_value", _freeze_value(self.invalid_value))

    def __str__(self) -> str:
        context = []
        for name in ("pool_id", "team_id", "variant_id", "slot_id", "field_name"):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        if self.invalid_value is not None:
            context.append(f"invalid_value={self.invalid_value!r}")
        suffix = f" [{', '.join(context)}]" if context else ""
        return f"{self.path}: {self.message}{suffix}"


class TeamPoolError(Exception):
    """Base class for team-pool errors."""


class TeamPoolValidationError(TeamPoolError):
    """All independently detectable validation issues for one document."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("TeamPoolValidationError requires at least one issue")
        super().__init__("\n".join(str(issue) for issue in self.issues))


class TeamPoolRegistryError(TeamPoolError):
    """An immutable registry could not be built unambiguously."""
