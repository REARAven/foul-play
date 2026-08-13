"""Sanitized errors for external Blind Ladder registry configuration."""

from __future__ import annotations


class BlindPoolValidationError(ValueError):
    """A fail-fast configuration, registry, or integrity validation error."""

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
