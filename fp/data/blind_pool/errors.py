"""Sanitized errors for external Blind Ladder configuration and state."""

from __future__ import annotations


class BlindPoolValidationError(ValueError):
    """A fail-fast configuration, registry, integrity, or state error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        team_id: str | None = None,
    ) -> None:
        self.code = code
        self.team_id = team_id
        context = "" if team_id is None else " team_id={}".format(team_id)
        super().__init__("{}: {}{}".format(code, message, context))


class BlindPoolLifecycleError(RuntimeError):
    """A sanitized challenge-lifecycle failure with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__("{}: {}".format(code, message))


class BlindPoolReconciliationRequired(BlindPoolLifecycleError):
    """An ambiguous post-accept outcome requiring an explicit decision."""
