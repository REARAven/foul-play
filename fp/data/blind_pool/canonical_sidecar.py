"""Strict structural parsing for trusted canonical Foul Play sidecars."""

from __future__ import annotations

import re
from typing import Any, cast

from .canonical_models import (
    CANONICAL_FORMAT_ID,
    CANONICAL_SIDECAR_SCHEMA_VERSION,
    CANONICAL_STAT_KEYS,
    CanonicalBattleTeamRecord,
    ParsedCanonicalSidecar,
    _fail,
    _require_exact_fields,
    _require_opaque_team_id,
    _strict_json_bytes,
)


_TOP_LEVEL_FIELDS = frozenset({"schema_version", "team_id", "format_id", "sets"})
_SET_FIELDS = frozenset(
    {
        "slot",
        "name",
        "species",
        "species_id",
        "item",
        "item_id",
        "ability",
        "ability_id",
        "moves",
        "nature",
        "nature_id",
        "evs",
        "ivs",
        "gender",
        "level",
        "happiness",
        "shiny",
        "hidden_power_type",
        "hidden_power_type_id",
        "pokeball",
        "pokeball_id",
        "gigantamax",
        "dynamax_level",
        "tera_type",
        "tera_type_id",
    }
)
_MOVE_FIELDS = frozenset({"slot", "name", "id"})
_STAT_FIELDS = frozenset(CANONICAL_STAT_KEYS)
_CANONICAL_ID_PATTERN = re.compile(r"^[a-z0-9]+$")
_GENDERS = frozenset({"", "M", "F", "N"})


def _require_string(
    value: Any,
    *,
    allow_empty: bool,
    team_id: str,
) -> str:
    if not isinstance(value, str) or "\x00" in value or (not allow_empty and not value):
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar text is invalid",
            team_id=team_id,
        )
    return value


def _require_canonical_id(value: Any, *, team_id: str) -> str:
    if not isinstance(value, str) or _CANONICAL_ID_PATTERN.fullmatch(value) is None:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar ID is invalid",
            team_id=team_id,
        )
    return value


def _require_nullable_pair(
    display: Any,
    canonical_id: Any,
    *,
    team_id: str,
) -> tuple[str | None, str | None]:
    if display is None or canonical_id is None:
        if display is not None or canonical_id is not None:
            _fail(
                "CANONICAL_SIDECAR_INVALID",
                "Canonical sidecar nullable pair is invalid",
                team_id=team_id,
            )
        return None, None
    return (
        _require_string(display, allow_empty=False, team_id=team_id),
        _require_canonical_id(canonical_id, team_id=team_id),
    )


def _require_integer(
    value: Any,
    minimum: int,
    maximum: int,
    *,
    team_id: str,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar integer is invalid",
            team_id=team_id,
        )
    return value


def _parse_stats(
    value: Any,
    *,
    maximum: int,
    enforce_ev_total: bool,
    team_id: str,
) -> tuple[int, int, int, int, int, int]:
    stats = _require_exact_fields(
        value,
        _STAT_FIELDS,
        code="CANONICAL_SIDECAR_INVALID",
        context="Canonical sidecar stats",
        team_id=team_id,
    )
    parsed = tuple(
        _require_integer(stats[key], 0, maximum, team_id=team_id)
        for key in CANONICAL_STAT_KEYS
    )
    if enforce_ev_total and sum(parsed) > 510:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar EV total is invalid",
            team_id=team_id,
        )
    return cast(tuple[int, int, int, int, int, int], parsed)


def _validate_moves(value: Any, *, team_id: str) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar move count is invalid",
            team_id=team_id,
        )
    for expected_slot, raw_move in enumerate(value, start=1):
        move = _require_exact_fields(
            raw_move,
            _MOVE_FIELDS,
            code="CANONICAL_SIDECAR_INVALID",
            context="Canonical sidecar move",
            team_id=team_id,
        )
        if type(move["slot"]) is not int or move["slot"] != expected_slot:
            _fail(
                "CANONICAL_SIDECAR_INVALID",
                "Canonical sidecar move slot is invalid",
                team_id=team_id,
            )
        _require_string(move["name"], allow_empty=False, team_id=team_id)
        _require_canonical_id(move["id"], team_id=team_id)


def _parse_set(
    value: Any,
    *,
    expected_slot: int,
    team_id: str,
) -> CanonicalBattleTeamRecord:
    canonical_set = _require_exact_fields(
        value,
        _SET_FIELDS,
        code="CANONICAL_SIDECAR_INVALID",
        context="Canonical sidecar set",
        team_id=team_id,
    )
    if type(canonical_set["slot"]) is not int or canonical_set["slot"] != expected_slot:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar set slot is invalid",
            team_id=team_id,
        )

    _require_string(canonical_set["name"], allow_empty=True, team_id=team_id)
    _require_string(canonical_set["species"], allow_empty=False, team_id=team_id)
    species_id = _require_canonical_id(canonical_set["species_id"], team_id=team_id)
    _require_nullable_pair(
        canonical_set["item"],
        canonical_set["item_id"],
        team_id=team_id,
    )
    _require_string(canonical_set["ability"], allow_empty=False, team_id=team_id)
    _require_canonical_id(canonical_set["ability_id"], team_id=team_id)
    _validate_moves(canonical_set["moves"], team_id=team_id)
    _require_string(canonical_set["nature"], allow_empty=False, team_id=team_id)
    nature_id = _require_canonical_id(canonical_set["nature_id"], team_id=team_id)
    evs = _parse_stats(
        canonical_set["evs"],
        maximum=252,
        enforce_ev_total=True,
        team_id=team_id,
    )
    ivs = _parse_stats(
        canonical_set["ivs"],
        maximum=31,
        enforce_ev_total=False,
        team_id=team_id,
    )
    gender = _require_string(
        canonical_set["gender"],
        allow_empty=True,
        team_id=team_id,
    )
    if gender not in _GENDERS:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar gender is invalid",
            team_id=team_id,
        )
    _require_integer(canonical_set["level"], 1, 100, team_id=team_id)
    _require_integer(canonical_set["happiness"], 0, 255, team_id=team_id)
    if type(canonical_set["shiny"]) is not bool:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar shiny flag is invalid",
            team_id=team_id,
        )
    _require_nullable_pair(
        canonical_set["hidden_power_type"],
        canonical_set["hidden_power_type_id"],
        team_id=team_id,
    )
    _require_nullable_pair(
        canonical_set["pokeball"],
        canonical_set["pokeball_id"],
        team_id=team_id,
    )
    if canonical_set["gigantamax"] is not False:
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar Gigantamax state is invalid",
            team_id=team_id,
        )
    if (
        type(canonical_set["dynamax_level"]) is not int
        or canonical_set["dynamax_level"] != 10
    ):
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar Dynamax state is invalid",
            team_id=team_id,
        )
    if (
        canonical_set["tera_type"] is not None
        or canonical_set["tera_type_id"] is not None
    ):
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar Tera state is invalid",
            team_id=team_id,
        )
    return CanonicalBattleTeamRecord(
        species_id=species_id,
        nature_id=nature_id,
        evs=evs,
        ivs=ivs,
    )


def parse_canonical_sidecar(
    raw: bytes,
    *,
    expected_team_id: str,
) -> ParsedCanonicalSidecar:
    """Parse one integrity-verified sidecar without Pokémon interpretation."""

    team_id = _require_opaque_team_id(
        expected_team_id,
        code="CANONICAL_SIDECAR_INVALID",
    )
    document = _strict_json_bytes(
        raw,
        encoding_code="CANONICAL_SIDECAR_INVALID",
        duplicate_code="CANONICAL_SIDECAR_INVALID",
        json_code="CANONICAL_SIDECAR_INVALID",
        team_id=team_id,
    )
    sidecar = _require_exact_fields(
        document,
        _TOP_LEVEL_FIELDS,
        code="CANONICAL_SIDECAR_INVALID",
        context="Canonical sidecar",
        team_id=team_id,
    )
    if (
        type(sidecar["schema_version"]) is not int
        or sidecar["schema_version"] != CANONICAL_SIDECAR_SCHEMA_VERSION
        or sidecar["team_id"] != team_id
        or sidecar["format_id"] != CANONICAL_FORMAT_ID
        or not isinstance(sidecar["sets"], list)
        or len(sidecar["sets"]) != 6
    ):
        _fail(
            "CANONICAL_SIDECAR_INVALID",
            "Canonical sidecar binding or shape is invalid",
            team_id=team_id,
        )
    records = tuple(
        _parse_set(raw_set, expected_slot=index, team_id=team_id)
        for index, raw_set in enumerate(sidecar["sets"], start=1)
    )
    return ParsedCanonicalSidecar(
        schema_version=CANONICAL_SIDECAR_SCHEMA_VERSION,
        team_id=team_id,
        format_id=CANONICAL_FORMAT_ID,
        records=records,
    )
