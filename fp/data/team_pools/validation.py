"""Aggregating validation for versioned complete-team-pool documents.

Validation uses the currently loaded Pokédex and move registries; callers must
apply the intended format overlay first.  The repository has no authoritative
complete item registry or learnset registry, so v1 proves item/no-item syntax
but not item existence, and deliberately does not guess move learnset legality.
Abilities are checked against the selected species' currently loaded Pokédex
entry.  Loading never mutates any of those registries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from fp.battle.helpers import natures, normalize_name
from fp.data import all_move_json, pokedex

from .errors import TeamPoolValidationError, ValidationIssue
from .models import (
    PUBLIC_FIELD_NAMES,
    SCHEMA_VERSION,
    STAT_KEYS,
    PokemonRecord,
    PoolIdentity,
    SourceLocation,
    StatValues,
    TeamPool,
    TeamRecord,
    TeamRecordId,
    canonical_roster_key,
)


class JsonObjectPairs(list[tuple[str, Any]]):
    """JSON object representation that retains duplicate source keys."""


_TOP_FIELDS = frozenset({"schema_version", "pool", "teams"})
_POOL_FIELDS = frozenset(
    {
        "pool_id",
        "pool_version",
        "format_id",
        "patch_version",
        "source_documents",
        "display_name",
        "default_public_fields",
        "metadata",
    }
)
_TEAM_FIELDS = frozenset(
    {
        "team_id",
        "variant_id",
        "variant_of",
        "display_name",
        "roster_key",
        "pokemon",
        "metadata",
    }
)
_POKEMON_FIELDS = frozenset(
    {
        "slot_id",
        "species_id",
        "base_species_id",
        "item_id",
        "base_ability_id",
        "current_ability_id",
        "move_ids",
        "nature_id",
        "evs",
        "ivs",
        "level",
        "public_fields",
        "metadata",
    }
)
_VARIANT_REF_FIELDS = frozenset({"team_id", "variant_id"})
_MISSING = object()


def _issue(
    issues: list[ValidationIssue],
    path: str,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
    field_name: str | None = None,
    invalid_value: Any = None,
) -> None:
    context = context or {}
    issues.append(
        ValidationIssue(
            path=path,
            message=message,
            pool_id=context.get("pool_id"),
            team_id=context.get("team_id"),
            variant_id=context.get("variant_id"),
            slot_id=context.get("slot_id"),
            field_name=field_name,
            invalid_value=invalid_value,
        )
    )


def _materialize(value: Any, path: str, issues: list[ValidationIssue]) -> Any:
    if isinstance(value, JsonObjectPairs):
        result = {}
        seen = set()
        for key, item in value:
            key_path = f"{path}.{key}" if isinstance(key, str) else path
            if key in seen:
                _issue(
                    issues,
                    key_path,
                    f"duplicate JSON field {key!r}",
                    field_name=key if isinstance(key, str) else None,
                )
                continue
            seen.add(key)
            result[key] = _materialize(item, key_path, issues)
        return result
    if isinstance(value, dict):
        return {key: _materialize(item, f"{path}.{key}", issues) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, f"{path}[{index}]", issues) for index, item in enumerate(value)]
    return value


def _require_object(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        _issue(
            issues,
            path,
            "expected an object",
            context=context,
            invalid_value=value,
        )
        return None
    return value


def _check_fields(
    obj: dict[str, Any],
    allowed: frozenset[str],
    required: tuple[str, ...],
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any] | None = None,
) -> None:
    for name in required:
        if name not in obj:
            _issue(
                issues,
                f"{path}.{name}",
                "missing required field",
                context=context,
                field_name=name,
            )
    for name in obj:
        if name not in allowed:
            _issue(
                issues,
                f"{path}.{name}",
                "unexpected field",
                context=context,
                field_name=name,
                invalid_value=name,
            )


def _string(
    obj: dict[str, Any],
    name: str,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
    *,
    required: bool = True,
    nullable: bool = False,
    canonical: bool = False,
) -> str | None:
    value = obj.get(name, _MISSING)
    if value is _MISSING:
        return None
    field_path = f"{path}.{name}"
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        _issue(
            issues,
            field_path,
            "expected a string" if not nullable else "expected a string or null",
            context=context,
            field_name=name,
            invalid_value=value,
        )
        return None
    if required and not value:
        _issue(
            issues,
            field_path,
            "must not be empty",
            context=context,
            field_name=name,
            invalid_value=value,
        )
        return None
    if canonical and value and normalize_name(value) != value:
        _issue(
            issues,
            field_path,
            f"expected canonical ID {normalize_name(value)!r}, received {value!r}",
            context=context,
            field_name=name,
            invalid_value=value,
        )
        return None
    return value


def _metadata(
    obj: dict[str, Any], path: str, issues: list[ValidationIssue], context: Mapping[str, Any]
) -> dict[str, Any]:
    value = obj.get("metadata", {})
    if not isinstance(value, dict):
        _issue(
            issues,
            f"{path}.metadata",
            "expected an object",
            context=context,
            field_name="metadata",
            invalid_value=value,
        )
        return {}
    return value


def _public_fields(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        _issue(issues, path, "expected an array", context=context, invalid_value=value)
        return None
    result = []
    seen = set()
    for index, field_name in enumerate(value):
        field_path = f"{path}[{index}]"
        if not isinstance(field_name, str):
            _issue(
                issues,
                field_path,
                "expected a public-field name string",
                context=context,
                invalid_value=field_name,
            )
        elif field_name not in PUBLIC_FIELD_NAMES:
            _issue(
                issues,
                field_path,
                f"unrecognized public field {field_name!r}",
                context=context,
                invalid_value=field_name,
            )
        elif field_name in seen:
            _issue(
                issues,
                field_path,
                f"duplicate public field {field_name!r}",
                context=context,
                invalid_value=field_name,
            )
        else:
            seen.add(field_name)
            result.append(field_name)
    return tuple(result)


def _canonical_id_list(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
    *,
    expected_length: int | None = None,
) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        _issue(issues, path, "expected an array", context=context, invalid_value=value)
        return None
    if expected_length is not None and len(value) != expected_length:
        _issue(
            issues,
            path,
            f"expected exactly {expected_length} entries, received {len(value)}",
            context=context,
            invalid_value=value,
        )
    result = []
    valid = True
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str) or not item:
            _issue(
                issues,
                item_path,
                "expected a nonempty ID string",
                context=context,
                invalid_value=item,
            )
            valid = False
        elif normalize_name(item) != item:
            _issue(
                issues,
                item_path,
                f"expected canonical ID {normalize_name(item)!r}, received {item!r}",
                context=context,
                invalid_value=item,
            )
            valid = False
        else:
            result.append(item)
    return tuple(result) if valid else None


def _stat_values(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
    *,
    maximum: int,
    total_maximum: int | None,
) -> StatValues | None:
    obj = _require_object(value, path, issues, context=context)
    if obj is None:
        return None
    actual_keys = set(obj)
    expected_keys = set(STAT_KEYS)
    for name in STAT_KEYS:
        if name not in obj:
            _issue(
                issues,
                f"{path}.{name}",
                "missing required stat",
                context=context,
                field_name=name,
            )
    for name in sorted(actual_keys - expected_keys):
        _issue(
            issues,
            f"{path}.{name}",
            "unexpected stat field",
            context=context,
            field_name=name,
            invalid_value=obj[name],
        )
    values: dict[str, int] = {}
    for name in STAT_KEYS:
        value_for_stat = obj.get(name, _MISSING)
        if value_for_stat is _MISSING:
            continue
        if isinstance(value_for_stat, bool) or not isinstance(value_for_stat, int):
            _issue(
                issues,
                f"{path}.{name}",
                "expected an integer",
                context=context,
                field_name=name,
                invalid_value=value_for_stat,
            )
        elif not 0 <= value_for_stat <= maximum:
            _issue(
                issues,
                f"{path}.{name}",
                f"must be between 0 and {maximum}",
                context=context,
                field_name=name,
                invalid_value=value_for_stat,
            )
        else:
            values[name] = value_for_stat
    if total_maximum is not None and len(values) == 6 and sum(values.values()) > total_maximum:
        _issue(
            issues,
            path,
            f"total must not exceed {total_maximum}",
            context=context,
            invalid_value=sum(values.values()),
        )
    if actual_keys != expected_keys or len(values) != 6:
        return None
    return StatValues(
        hp=values["hp"],
        atk=values["atk"],
        defense=values["def"],
        spa=values["spa"],
        spd=values["spd"],
        spe=values["spe"],
    )


def _pokemon_record(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    pool_context: Mapping[str, Any],
    source_path: str,
) -> tuple[PokemonRecord | None, str | None]:
    obj = _require_object(value, path, issues, context=pool_context)
    if obj is None:
        return None, None
    provisional_slot = obj.get("slot_id") if isinstance(obj.get("slot_id"), str) else None
    context = dict(pool_context, slot_id=provisional_slot)
    start_count = len(issues)
    _check_fields(
        obj,
        _POKEMON_FIELDS,
        (
            "slot_id",
            "species_id",
            "item_id",
            "base_ability_id",
            "current_ability_id",
            "move_ids",
            "nature_id",
            "evs",
            "ivs",
            "level",
        ),
        path,
        issues,
        context,
    )
    slot_id = _string(obj, "slot_id", path, issues, context, canonical=True)
    species_id = _string(obj, "species_id", path, issues, context, canonical=True)
    item_id = _string(obj, "item_id", path, issues, context, canonical=True)
    base_ability_id = _string(obj, "base_ability_id", path, issues, context, canonical=True)
    current_ability_id = _string(
        obj, "current_ability_id", path, issues, context, canonical=True
    )
    nature_id = _string(obj, "nature_id", path, issues, context, canonical=True)
    explicit_base = _string(
        obj,
        "base_species_id",
        path,
        issues,
        context,
        required=False,
        canonical=True,
    )
    move_ids = _canonical_id_list(
        obj.get("move_ids", _MISSING),
        f"{path}.move_ids",
        issues,
        context,
        expected_length=4,
    )
    evs = _stat_values(
        obj.get("evs", _MISSING),
        f"{path}.evs",
        issues,
        context,
        maximum=252,
        total_maximum=510,
    )
    ivs = _stat_values(
        obj.get("ivs", _MISSING),
        f"{path}.ivs",
        issues,
        context,
        maximum=31,
        total_maximum=None,
    )
    level = obj.get("level", _MISSING)
    if level is not _MISSING and (isinstance(level, bool) or not isinstance(level, int)):
        _issue(
            issues,
            f"{path}.level",
            "expected an integer",
            context=context,
            field_name="level",
            invalid_value=level,
        )
        level = None
    elif level is not _MISSING and not 1 <= level <= 100:
        _issue(
            issues,
            f"{path}.level",
            "must be between 1 and 100",
            context=context,
            field_name="level",
            invalid_value=level,
        )
        level = None
    elif level is _MISSING:
        level = None

    public_fields = _public_fields(
        obj.get("public_fields", []), f"{path}.public_fields", issues, context
    )
    metadata = _metadata(obj, path, issues, context)

    derived_base = None
    species_data = None
    if species_id is not None:
        species_data = pokedex.get(species_id)
        if species_data is None:
            _issue(
                issues,
                f"{path}.species_id",
                f"unknown exact species/form ID {species_id!r}",
                context=context,
                field_name="species_id",
                invalid_value=species_id,
            )
        else:
            raw_base = species_data.get("baseSpecies")
            if raw_base is None:
                derived_base = species_id
            elif isinstance(raw_base, str):
                candidate = normalize_name(raw_base)
                if candidate in pokedex:
                    derived_base = candidate
                else:
                    _issue(
                        issues,
                        f"{path}.base_species_id",
                        f"Pokédex base species {candidate!r} cannot be resolved unambiguously",
                        context=context,
                        field_name="base_species_id",
                        invalid_value=raw_base,
                    )
            else:
                _issue(
                    issues,
                    f"{path}.base_species_id",
                    "Pokédex baseSpecies metadata is not a string",
                    context=context,
                    field_name="base_species_id",
                    invalid_value=raw_base,
                )
    if explicit_base is not None and derived_base is not None and explicit_base != derived_base:
        _issue(
            issues,
            f"{path}.base_species_id",
            f"must match authoritative base species {derived_base!r}",
            context=context,
            field_name="base_species_id",
            invalid_value=explicit_base,
        )

    if base_ability_id is not None:
        if species_data is not None:
            ability_data = species_data.get("abilities", {})
            known_abilities = (
                {normalize_name(ability) for ability in ability_data.values()}
                if isinstance(ability_data, dict)
                else set()
            )
            if base_ability_id not in known_abilities:
                _issue(
                    issues,
                    f"{path}.base_ability_id",
                    f"ability {base_ability_id!r} is not listed for species {species_id!r}",
                    context=context,
                    field_name="base_ability_id",
                    invalid_value=base_ability_id,
                )
        if current_ability_id is not None and base_ability_id != current_ability_id:
            _issue(
                issues,
                f"{path}.current_ability_id",
                "must equal base_ability_id in an initial schema-v1 team record",
                context=context,
                field_name="current_ability_id",
                invalid_value=current_ability_id,
            )
    if move_ids is not None:
        duplicates = sorted(move for move, count in Counter(move_ids).items() if count > 1)
        for move_id in duplicates:
            _issue(
                issues,
                f"{path}.move_ids",
                f"duplicate move ID {move_id!r}",
                context=context,
                field_name="move_ids",
                invalid_value=move_id,
            )
        for index, move_id in enumerate(move_ids):
            if move_id not in all_move_json:
                _issue(
                    issues,
                    f"{path}.move_ids[{index}]",
                    f"unknown move ID {move_id!r}",
                    context=context,
                    field_name="move_ids",
                    invalid_value=move_id,
                )
    if nature_id is not None and nature_id not in natures:
        _issue(
            issues,
            f"{path}.nature_id",
            f"unknown nature ID {nature_id!r}",
            context=context,
            field_name="nature_id",
            invalid_value=nature_id,
        )

    if len(issues) != start_count:
        return None, derived_base
    assert all(
        value is not None
        for value in (
            slot_id,
            species_id,
            derived_base,
            item_id,
            base_ability_id,
            current_ability_id,
            move_ids,
            nature_id,
            evs,
            ivs,
            level,
            public_fields,
        )
    )
    return (
        PokemonRecord(
            slot_id=slot_id,
            species_id=species_id,
            base_species_id=derived_base,
            item_id=item_id,
            base_ability_id=base_ability_id,
            current_ability_id=current_ability_id,
            move_ids=move_ids,  # type: ignore[arg-type]
            nature_id=nature_id,
            evs=evs,
            ivs=ivs,
            level=level,
            public_fields=public_fields,
            metadata=metadata,
            source_location=SourceLocation(source_path, path),
        ),
        derived_base,
    )


def _variant_reference(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
) -> TeamRecordId | None:
    if value is None:
        return None
    obj = _require_object(value, path, issues, context=context)
    if obj is None:
        return None
    start_count = len(issues)
    _check_fields(
        obj,
        _VARIANT_REF_FIELDS,
        ("team_id", "variant_id"),
        path,
        issues,
        context,
    )
    team_id = _string(obj, "team_id", path, issues, context, canonical=True)
    variant_id = _string(obj, "variant_id", path, issues, context, canonical=True)
    if len(issues) != start_count or team_id is None or variant_id is None:
        return None
    return TeamRecordId(team_id, variant_id)


def _team_record(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    pool_context: Mapping[str, Any],
    source_path: str,
) -> tuple[TeamRecord | None, TeamRecordId | None, TeamRecordId | None]:
    obj = _require_object(value, path, issues, context=pool_context)
    if obj is None:
        return None, None, None
    team_id_hint = obj.get("team_id") if isinstance(obj.get("team_id"), str) else None
    variant_id_hint = obj.get("variant_id") if isinstance(obj.get("variant_id"), str) else None
    context = dict(pool_context, team_id=team_id_hint, variant_id=variant_id_hint)
    start_count = len(issues)
    _check_fields(
        obj,
        _TEAM_FIELDS,
        ("team_id", "variant_id", "pokemon"),
        path,
        issues,
        context,
    )
    team_id = _string(obj, "team_id", path, issues, context, canonical=True)
    variant_id = _string(obj, "variant_id", path, issues, context, canonical=True)
    record_id = (
        TeamRecordId(team_id, variant_id)
        if team_id is not None and variant_id is not None
        else None
    )
    variant_of = _variant_reference(
        obj.get("variant_of"), f"{path}.variant_of", issues, context
    )
    display_name = _string(
        obj,
        "display_name",
        path,
        issues,
        context,
        required=False,
        nullable=True,
    )
    metadata = _metadata(obj, path, issues, context)

    pokemon_values = obj.get("pokemon", _MISSING)
    pokemon_records = []
    base_species_ids = []
    if not isinstance(pokemon_values, list):
        _issue(
            issues,
            f"{path}.pokemon",
            "expected an array",
            context=context,
            field_name="pokemon",
            invalid_value=pokemon_values,
        )
    else:
        if len(pokemon_values) != 6:
            _issue(
                issues,
                f"{path}.pokemon",
                f"expected exactly 6 Pokémon, received {len(pokemon_values)}",
                context=context,
                field_name="pokemon",
                invalid_value=len(pokemon_values),
            )
        for index, pokemon_value in enumerate(pokemon_values):
            record, base_species_id = _pokemon_record(
                pokemon_value,
                f"{path}.pokemon[{index}]",
                issues,
                context,
                source_path,
            )
            if record is not None:
                pokemon_records.append(record)
            if base_species_id is not None:
                base_species_ids.append(base_species_id)

    slot_ids = [record.slot_id for record in pokemon_records]
    for slot_id, count in sorted(Counter(slot_ids).items()):
        if count > 1:
            _issue(
                issues,
                f"{path}.pokemon",
                f"duplicate slot ID {slot_id!r}",
                context=context,
                field_name="slot_id",
                invalid_value=slot_id,
            )
    for base_species_id, count in sorted(Counter(base_species_ids).items()):
        if count > 1:
            _issue(
                issues,
                f"{path}.pokemon",
                f"Species Clause violation for base species {base_species_id!r}",
                context=context,
                field_name="base_species_id",
                invalid_value=base_species_id,
            )

    computed_roster_key = None
    if len(pokemon_records) == 6:
        computed_roster_key = canonical_roster_key(
            record.species_id for record in pokemon_records
        )
    if "roster_key" in obj:
        supplied_roster_key = _canonical_id_list(
            obj["roster_key"],
            f"{path}.roster_key",
            issues,
            context,
            expected_length=6,
        )
        if supplied_roster_key is not None and len(supplied_roster_key) == 6:
            if tuple(sorted(supplied_roster_key)) != supplied_roster_key:
                _issue(
                    issues,
                    f"{path}.roster_key",
                    "source roster key must be sorted canonically",
                    context=context,
                    field_name="roster_key",
                    invalid_value=supplied_roster_key,
                )
            elif computed_roster_key is not None and supplied_roster_key != computed_roster_key:
                _issue(
                    issues,
                    f"{path}.roster_key",
                    "source roster key does not match the six Pokémon records",
                    context=context,
                    field_name="roster_key",
                    invalid_value=supplied_roster_key,
                )

    if len(issues) != start_count or record_id is None or computed_roster_key is None:
        return None, record_id, variant_of
    return (
        TeamRecord(
            record_id=record_id,
            variant_of=variant_of,
            display_name=display_name,
            pokemon=tuple(pokemon_records),  # type: ignore[arg-type]
            roster_key=computed_roster_key,
            metadata=metadata,
            source_location=SourceLocation(source_path, path),
        ),
        record_id,
        variant_of,
    )


def validate_team_pool_document(document: Any, source_path: str = "<memory>") -> TeamPool:
    """Validate one parsed JSON value and return an immutable v1 pool.

    All independently detectable source problems are raised together as one
    :class:`TeamPoolValidationError`.
    """

    issues: list[ValidationIssue] = []
    document = _materialize(document, "$", issues)
    top = _require_object(document, "$", issues)
    if top is None:
        raise TeamPoolValidationError(issues)
    _check_fields(top, _TOP_FIELDS, ("schema_version", "pool", "teams"), "$", issues)
    schema_version = top.get("schema_version", _MISSING)
    if schema_version is not _MISSING and (
        isinstance(schema_version, bool) or not isinstance(schema_version, int)
    ):
        _issue(
            issues,
            "$.schema_version",
            "expected an integer",
            field_name="schema_version",
            invalid_value=schema_version,
        )
        raise TeamPoolValidationError(issues)
    if schema_version is not _MISSING and schema_version != SCHEMA_VERSION:
        _issue(
            issues,
            "$.schema_version",
            f"unsupported schema version {schema_version!r}; only version {SCHEMA_VERSION} is supported",
            field_name="schema_version",
            invalid_value=schema_version,
        )
        raise TeamPoolValidationError(issues)

    pool_obj = _require_object(top.get("pool", _MISSING), "$.pool", issues)
    if pool_obj is None:
        raise TeamPoolValidationError(issues)
    _check_fields(
        pool_obj,
        _POOL_FIELDS,
        ("pool_id", "pool_version", "format_id"),
        "$.pool",
        issues,
    )
    pool_id_hint = pool_obj.get("pool_id") if isinstance(pool_obj.get("pool_id"), str) else None
    pool_context = {"pool_id": pool_id_hint}
    pool_id = _string(pool_obj, "pool_id", "$.pool", issues, pool_context, canonical=True)
    pool_version = _string(pool_obj, "pool_version", "$.pool", issues, pool_context)
    format_id = _string(pool_obj, "format_id", "$.pool", issues, pool_context, canonical=True)
    patch_version = _string(
        pool_obj,
        "patch_version",
        "$.pool",
        issues,
        pool_context,
        nullable=True,
        required=False,
    )
    display_name = _string(
        pool_obj,
        "display_name",
        "$.pool",
        issues,
        pool_context,
        nullable=True,
        required=False,
    )
    metadata = _metadata(pool_obj, "$.pool", issues, pool_context)
    default_public_fields = _public_fields(
        pool_obj.get("default_public_fields", []),
        "$.pool.default_public_fields",
        issues,
        pool_context,
    )
    source_documents = _canonical_id_list(
        pool_obj.get("source_documents", []),
        "$.pool.source_documents",
        issues,
        pool_context,
    )
    if source_documents is not None:
        duplicates = sorted(
            source_id for source_id, count in Counter(source_documents).items() if count > 1
        )
        for source_id in duplicates:
            _issue(
                issues,
                "$.pool.source_documents",
                f"duplicate source-document ID {source_id!r}",
                context=pool_context,
                field_name="source_documents",
                invalid_value=source_id,
            )

    teams_value = top.get("teams", _MISSING)
    team_records = []
    record_ids = []
    references: list[tuple[str, TeamRecordId, TeamRecordId]] = []
    if not isinstance(teams_value, list):
        _issue(issues, "$.teams", "expected an array", invalid_value=teams_value)
    else:
        if not teams_value:
            _issue(issues, "$.teams", "pool must contain at least one team")
        for index, team_value in enumerate(teams_value):
            team, record_id, variant_of = _team_record(
                team_value,
                f"$.teams[{index}]",
                issues,
                pool_context,
                source_path,
            )
            if team is not None:
                team_records.append(team)
            if record_id is not None:
                record_ids.append(record_id)
                if variant_of is not None:
                    references.append((f"$.teams[{index}].variant_of", record_id, variant_of))

    known_record_ids = set(record_ids)
    for record_id, count in sorted(Counter(record_ids).items()):
        if count > 1:
            _issue(
                issues,
                "$.teams",
                f"duplicate team-record identity {record_id!r}",
                context=pool_context,
                invalid_value=record_id,
            )
    for path, record_id, reference in references:
        if reference == record_id:
            _issue(
                issues,
                path,
                "variant_of must not reference the record itself",
                context={
                    **pool_context,
                    "team_id": record_id.team_id,
                    "variant_id": record_id.variant_id,
                },
                field_name="variant_of",
                invalid_value=reference,
            )
        elif reference not in known_record_ids:
            _issue(
                issues,
                path,
                f"variant_of references unknown team record {reference!r}",
                context={
                    **pool_context,
                    "team_id": record_id.team_id,
                    "variant_id": record_id.variant_id,
                },
                field_name="variant_of",
                invalid_value=reference,
            )

    if issues:
        raise TeamPoolValidationError(issues)
    assert all(
        value is not None
        for value in (
            schema_version,
            pool_id,
            pool_version,
            format_id,
            default_public_fields,
            source_documents,
        )
    )
    return TeamPool(
        schema_version=schema_version,
        identity=PoolIdentity(pool_id, pool_version, format_id),
        patch_version=patch_version,
        source_documents=source_documents,
        display_name=display_name,
        default_public_fields=default_public_fields,
        metadata=metadata,
        teams=tuple(team_records),
    )
