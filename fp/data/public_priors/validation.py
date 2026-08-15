"""Aggregate validation for schema-v1 public opponent-prior documents.

Validation reads the currently loaded Pokédex and move registry without changing
them.  Callers must apply the correct format overlay first.  The repository has
no authoritative complete item or learnset registry, so schema v1 validates
item/no-item syntax but not item existence and does not guess learnset legality.
The ``visibility`` declaration is authorization metadata; a human or offline
import process must establish that the underlying material was genuinely public.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from fp.battle.helpers import natures, normalize_name
from fp.data import all_move_json, pokedex

from .errors import PublicPriorValidationError, ValidationIssue
from .models import (
    NO_ITEM_ID,
    PUBLIC_VISIBILITY,
    SCHEMA_VERSION,
    STAT_KEYS,
    PublicPriorDataset,
    PublicPriorIdentity,
    PublicSetVariant,
    PublicSource,
    PublicSourceKind,
    PublicStatValues,
    SpeciesPrior,
)


class JsonObjectPairs(list[tuple[str, Any]]):
    """JSON object representation that retains duplicate source keys."""


_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "visibility",
        "dataset_id",
        "dataset_version",
        "format_id",
        "patch_version",
        "display_name",
        "metadata",
        "sources",
        "species",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "kind",
        "public_date",
        "public_version",
        "description",
        "metadata",
    }
)
_SPECIES_FIELDS = frozenset({"species_id", "base_species_id", "variants", "metadata"})
_VARIANT_FIELDS = frozenset(
    {
        "variant_id",
        "weight",
        "item_id",
        "base_ability_id",
        "move_ids",
        "nature_id",
        "evs",
        "ivs",
        "level",
        "source_ids",
        "metadata",
    }
)
_MISSING = object()


def _issue(
    issues: list[ValidationIssue],
    path: str,
    explanation: str,
    *,
    context: Mapping[str, Any] | None = None,
    field: str | None = None,
    invalid_value: Any = None,
) -> None:
    context = context or {}
    issues.append(
        ValidationIssue(
            path=path,
            explanation=explanation,
            dataset_id=context.get("dataset_id"),
            species_id=context.get("species_id"),
            variant_id=context.get("variant_id"),
            field=field,
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
                    field=key if isinstance(key, str) else None,
                )
                continue
            seen.add(key)
            result[key] = _materialize(item, key_path, issues)
        return result
    if isinstance(value, dict):
        return {
            key: _materialize(item, f"{path}.{key}", issues)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _materialize(item, f"{path}[{index}]", issues)
            for index, item in enumerate(value)
        ]
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
                field=name,
            )
    for name in obj:
        if name not in allowed:
            _issue(
                issues,
                f"{path}.{name}",
                "unexpected field",
                context=context,
                field=name,
                invalid_value=obj[name],
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
            field=name,
            invalid_value=value,
        )
        return None
    if required and not value:
        _issue(
            issues,
            field_path,
            "must not be empty",
            context=context,
            field=name,
            invalid_value=value,
        )
        return None
    if canonical and value and normalize_name(value) != value:
        _issue(
            issues,
            field_path,
            f"expected canonical ID {normalize_name(value)!r}, received {value!r}",
            context=context,
            field=name,
            invalid_value=value,
        )
        return None
    return value


def _metadata(
    obj: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    value = obj.get("metadata", {})
    if not isinstance(value, dict):
        _issue(
            issues,
            f"{path}.metadata",
            "expected an object",
            context=context,
            field="metadata",
            invalid_value=value,
        )
        return {}
    return value


def _stat_values(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
    *,
    maximum: int,
    total_maximum: int | None,
) -> PublicStatValues | None:
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
                field=name,
            )
    for name in sorted(actual_keys - expected_keys):
        _issue(
            issues,
            f"{path}.{name}",
            "unexpected stat field",
            context=context,
            field=name,
            invalid_value=obj[name],
        )
    values: dict[str, int] = {}
    for name in STAT_KEYS:
        stat = obj.get(name, _MISSING)
        if stat is _MISSING:
            continue
        if isinstance(stat, bool) or not isinstance(stat, int):
            _issue(
                issues,
                f"{path}.{name}",
                "expected an integer",
                context=context,
                field=name,
                invalid_value=stat,
            )
        elif not 0 <= stat <= maximum:
            _issue(
                issues,
                f"{path}.{name}",
                f"must be between 0 and {maximum}",
                context=context,
                field=name,
                invalid_value=stat,
            )
        else:
            values[name] = stat
    if (
        total_maximum is not None
        and len(values) == 6
        and sum(values.values()) > total_maximum
    ):
        _issue(
            issues,
            path,
            f"total must not exceed {total_maximum}",
            context=context,
            invalid_value=sum(values.values()),
        )
    if actual_keys != expected_keys or len(values) != 6:
        return None
    return PublicStatValues(
        hp=values["hp"],
        atk=values["atk"],
        defense=values["def"],
        spa=values["spa"],
        spd=values["spd"],
        spe=values["spe"],
    )


def _source_record(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    dataset_context: Mapping[str, Any],
) -> tuple[PublicSource | None, str | None]:
    obj = _require_object(value, path, issues, context=dataset_context)
    if obj is None:
        return None, None
    _check_fields(
        obj,
        _SOURCE_FIELDS,
        ("source_id", "kind"),
        path,
        issues,
        dataset_context,
    )
    start_count = len(issues)
    source_id = _string(obj, "source_id", path, issues, dataset_context, canonical=True)
    kind_value = _string(obj, "kind", path, issues, dataset_context)
    kind = None
    if kind_value is not None:
        try:
            kind = PublicSourceKind(kind_value)
        except ValueError:
            _issue(
                issues,
                f"{path}.kind",
                "unsupported source kind; schema v1 permits only explicitly public kinds",
                context=dataset_context,
                field="kind",
                invalid_value=kind_value,
            )
    public_date = _string(
        obj,
        "public_date",
        path,
        issues,
        dataset_context,
        required=False,
        nullable=True,
    )
    public_version = _string(
        obj,
        "public_version",
        path,
        issues,
        dataset_context,
        required=False,
        nullable=True,
    )
    description = _string(
        obj,
        "description",
        path,
        issues,
        dataset_context,
        required=False,
        nullable=True,
    )
    metadata = _metadata(obj, path, issues, dataset_context)
    if len(issues) != start_count or source_id is None or kind is None:
        return None, source_id
    return (
        PublicSource(
            source_id=source_id,
            kind=kind,
            public_date=public_date,
            public_version=public_version,
            description=description,
            metadata=metadata,
        ),
        source_id,
    )


def _canonical_id_array(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    context: Mapping[str, Any],
    *,
    expected_length: int | None = None,
    require_nonempty: bool = False,
) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        _issue(
            issues,
            path,
            "expected an array",
            context=context,
            invalid_value=value,
        )
        return None
    if expected_length is not None and len(value) != expected_length:
        _issue(
            issues,
            path,
            f"expected exactly {expected_length} entries, received {len(value)}",
            context=context,
            invalid_value=value,
        )
    if require_nonempty and not value:
        _issue(issues, path, "must contain at least one entry", context=context)
    valid = True
    result = []
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


def _variant_record(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    species_context: Mapping[str, Any],
    known_source_ids: set[str],
    species_data: Mapping[str, Any] | None,
) -> tuple[PublicSetVariant | None, str | None]:
    obj = _require_object(value, path, issues, context=species_context)
    if obj is None:
        return None, None
    provisional_variant = (
        obj.get("variant_id") if isinstance(obj.get("variant_id"), str) else None
    )
    context = dict(species_context, variant_id=provisional_variant)
    start_count = len(issues)
    _check_fields(
        obj,
        _VARIANT_FIELDS,
        (
            "variant_id",
            "weight",
            "item_id",
            "base_ability_id",
            "move_ids",
            "nature_id",
            "evs",
            "ivs",
            "level",
            "source_ids",
        ),
        path,
        issues,
        context,
    )
    variant_id = _string(obj, "variant_id", path, issues, context, canonical=True)
    item_id = _string(obj, "item_id", path, issues, context, canonical=True)
    base_ability_id = _string(
        obj, "base_ability_id", path, issues, context, canonical=True
    )
    nature_id = _string(obj, "nature_id", path, issues, context, canonical=True)
    metadata = _metadata(obj, path, issues, context)

    weight = obj.get("weight", _MISSING)
    if weight is not _MISSING and (
        isinstance(weight, bool) or not isinstance(weight, (int, float))
    ):
        _issue(
            issues,
            f"{path}.weight",
            "expected a number",
            context=context,
            field="weight",
            invalid_value=weight,
        )
        weight = None
    elif weight is not _MISSING and (not math.isfinite(weight) or weight <= 0):
        _issue(
            issues,
            f"{path}.weight",
            "must be finite and strictly greater than zero",
            context=context,
            field="weight",
            invalid_value=weight,
        )
        weight = None
    elif weight is _MISSING:
        weight = None

    move_ids = _canonical_id_array(
        obj.get("move_ids", _MISSING),
        f"{path}.move_ids",
        issues,
        context,
        expected_length=4,
    )
    if move_ids is not None:
        for move_id, count in sorted(Counter(move_ids).items()):
            if count > 1:
                _issue(
                    issues,
                    f"{path}.move_ids",
                    f"duplicate move ID {move_id!r}",
                    context=context,
                    field="move_ids",
                    invalid_value=move_id,
                )
        for index, move_id in enumerate(move_ids):
            if move_id not in all_move_json:
                _issue(
                    issues,
                    f"{path}.move_ids[{index}]",
                    f"unknown move ID {move_id!r}",
                    context=context,
                    field="move_ids",
                    invalid_value=move_id,
                )

    if base_ability_id is not None and species_data is not None:
        raw_abilities = species_data.get("abilities", {})
        known_abilities = (
            {
                normalize_name(ability)
                for ability in raw_abilities.values()
                if isinstance(ability, str)
            }
            if isinstance(raw_abilities, dict)
            else set()
        )
        if base_ability_id not in known_abilities:
            _issue(
                issues,
                f"{path}.base_ability_id",
                "base ability does not belong to the exact species/form",
                context=context,
                field="base_ability_id",
                invalid_value=base_ability_id,
            )

    if nature_id is not None and nature_id not in natures:
        _issue(
            issues,
            f"{path}.nature_id",
            f"unknown nature ID {nature_id!r}",
            context=context,
            field="nature_id",
            invalid_value=nature_id,
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
    if level is not _MISSING and (
        isinstance(level, bool) or not isinstance(level, int)
    ):
        _issue(
            issues,
            f"{path}.level",
            "expected an integer",
            context=context,
            field="level",
            invalid_value=level,
        )
        level = None
    elif level is not _MISSING and not 1 <= level <= 100:
        _issue(
            issues,
            f"{path}.level",
            "must be between 1 and 100",
            context=context,
            field="level",
            invalid_value=level,
        )
        level = None
    elif level is _MISSING:
        level = None

    source_ids = _canonical_id_array(
        obj.get("source_ids", _MISSING),
        f"{path}.source_ids",
        issues,
        context,
        require_nonempty=True,
    )
    if source_ids is not None:
        for source_id, count in sorted(Counter(source_ids).items()):
            if count > 1:
                _issue(
                    issues,
                    f"{path}.source_ids",
                    f"duplicate source reference {source_id!r}",
                    context=context,
                    field="source_ids",
                    invalid_value=source_id,
                )
            if source_id not in known_source_ids:
                _issue(
                    issues,
                    f"{path}.source_ids",
                    f"unknown public source reference {source_id!r}",
                    context=context,
                    field="source_ids",
                    invalid_value=source_id,
                )

    if item_id is not None and item_id != NO_ITEM_ID:
        # The repository has no authoritative complete item registry.  A
        # nonempty canonical ID is therefore the strongest safe v1 check.
        pass

    if len(issues) != start_count or None in (
        variant_id,
        weight,
        item_id,
        base_ability_id,
        move_ids,
        nature_id,
        evs,
        ivs,
        level,
        source_ids,
    ):
        return None, variant_id
    return (
        PublicSetVariant(
            variant_id=variant_id,
            weight=weight,
            item_id=item_id,
            base_ability_id=base_ability_id,
            move_ids=move_ids,  # type: ignore[arg-type]
            nature_id=nature_id,
            evs=evs,
            ivs=ivs,
            level=level,
            source_ids=source_ids,
            metadata=metadata,
        ),
        variant_id,
    )


def _species_record(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    dataset_context: Mapping[str, Any],
    known_source_ids: set[str],
) -> tuple[SpeciesPrior | None, str | None]:
    obj = _require_object(value, path, issues, context=dataset_context)
    if obj is None:
        return None, None
    provisional_species = (
        obj.get("species_id") if isinstance(obj.get("species_id"), str) else None
    )
    context = dict(dataset_context, species_id=provisional_species)
    start_count = len(issues)
    _check_fields(
        obj,
        _SPECIES_FIELDS,
        ("species_id", "variants"),
        path,
        issues,
        context,
    )
    species_id = _string(obj, "species_id", path, issues, context, canonical=True)
    explicit_base = _string(
        obj,
        "base_species_id",
        path,
        issues,
        context,
        required=False,
        canonical=True,
    )
    metadata = _metadata(obj, path, issues, context)

    species_data = pokedex.get(species_id) if species_id is not None else None
    authoritative_base = None
    if species_id is not None:
        if species_data is None:
            _issue(
                issues,
                f"{path}.species_id",
                f"unknown exact species/form ID {species_id!r}",
                context=context,
                field="species_id",
                invalid_value=species_id,
            )
        else:
            raw_base = species_data.get("baseSpecies")
            if raw_base is None:
                authoritative_base = species_id
            elif isinstance(raw_base, str):
                authoritative_base = normalize_name(raw_base)
                if authoritative_base not in pokedex:
                    _issue(
                        issues,
                        f"{path}.base_species_id",
                        "Pokédex base species cannot be resolved authoritatively",
                        context=context,
                        field="base_species_id",
                        invalid_value=raw_base,
                    )
                    authoritative_base = None
            else:
                _issue(
                    issues,
                    f"{path}.base_species_id",
                    "Pokédex baseSpecies metadata is not a string",
                    context=context,
                    field="base_species_id",
                    invalid_value=raw_base,
                )
    if (
        explicit_base is not None
        and authoritative_base is not None
        and explicit_base != authoritative_base
    ):
        _issue(
            issues,
            f"{path}.base_species_id",
            f"must match authoritative base species {authoritative_base!r}",
            context=context,
            field="base_species_id",
            invalid_value=explicit_base,
        )

    variants_value = obj.get("variants", _MISSING)
    variants = []
    variant_ids = []
    if not isinstance(variants_value, list):
        _issue(
            issues,
            f"{path}.variants",
            "expected an array",
            context=context,
            field="variants",
            invalid_value=variants_value,
        )
    else:
        if not variants_value:
            _issue(
                issues,
                f"{path}.variants",
                "species must contain at least one public variant",
                context=context,
                field="variants",
            )
        for index, variant_value in enumerate(variants_value):
            variant, variant_id = _variant_record(
                variant_value,
                f"{path}.variants[{index}]",
                issues,
                context,
                known_source_ids,
                species_data,
            )
            if variant is not None:
                variants.append(variant)
            if variant_id is not None:
                variant_ids.append(variant_id)
        for variant_id, count in sorted(Counter(variant_ids).items()):
            if count > 1:
                _issue(
                    issues,
                    f"{path}.variants",
                    f"duplicate variant ID {variant_id!r}",
                    context=dict(context, variant_id=variant_id),
                    field="variant_id",
                    invalid_value=variant_id,
                )

    if len(issues) != start_count or species_id is None or not variants:
        return None, species_id
    return (
        SpeciesPrior(
            species_id=species_id,
            base_species_id=explicit_base,
            variants=tuple(variants),
            metadata=metadata,
        ),
        species_id,
    )


def validate_public_prior_document(
    document: Any, source_path: str = "<memory>"
) -> PublicPriorDataset:
    """Validate one parsed JSON value and return an immutable v1 dataset."""

    del (
        source_path
    )  # Diagnostics use JSON paths; the loader retains no live file object.
    issues: list[ValidationIssue] = []
    document = _materialize(document, "$", issues)
    top = _require_object(document, "$", issues)
    if top is None:
        raise PublicPriorValidationError(issues)
    _check_fields(
        top,
        _TOP_FIELDS,
        (
            "schema_version",
            "visibility",
            "dataset_id",
            "dataset_version",
            "format_id",
            "sources",
            "species",
        ),
        "$",
        issues,
    )

    schema_version = top.get("schema_version", _MISSING)
    if schema_version is not _MISSING and (
        isinstance(schema_version, bool) or not isinstance(schema_version, int)
    ):
        _issue(
            issues,
            "$.schema_version",
            "expected an integer",
            field="schema_version",
            invalid_value=schema_version,
        )
        raise PublicPriorValidationError(issues)
    if schema_version is not _MISSING and schema_version != SCHEMA_VERSION:
        _issue(
            issues,
            "$.schema_version",
            f"unsupported schema version {schema_version!r}; only version {SCHEMA_VERSION} is supported",
            field="schema_version",
            invalid_value=schema_version,
        )
        raise PublicPriorValidationError(issues)

    dataset_hint = (
        top.get("dataset_id") if isinstance(top.get("dataset_id"), str) else None
    )
    context = {"dataset_id": dataset_hint}
    visibility = _string(top, "visibility", "$", issues, context, required=False)
    if visibility is not None and visibility != PUBLIC_VISIBILITY:
        _issue(
            issues,
            "$.visibility",
            f"visibility must be exactly {PUBLIC_VISIBILITY!r}",
            context=context,
            field="visibility",
            invalid_value=visibility,
        )
    dataset_id = _string(top, "dataset_id", "$", issues, context, canonical=True)
    dataset_version = _string(top, "dataset_version", "$", issues, context)
    format_id = _string(top, "format_id", "$", issues, context, canonical=True)
    patch_version = _string(
        top,
        "patch_version",
        "$",
        issues,
        context,
        required=False,
        nullable=True,
    )
    display_name = _string(
        top,
        "display_name",
        "$",
        issues,
        context,
        required=False,
        nullable=True,
    )
    metadata = _metadata(top, "$", issues, context)

    sources_value = top.get("sources", _MISSING)
    sources = []
    source_ids = []
    if not isinstance(sources_value, list):
        _issue(
            issues,
            "$.sources",
            "expected an array",
            context=context,
            field="sources",
            invalid_value=sources_value,
        )
    else:
        if not sources_value:
            _issue(
                issues,
                "$.sources",
                "dataset must contain at least one public source",
                context=context,
                field="sources",
            )
        for index, source_value in enumerate(sources_value):
            source, source_id = _source_record(
                source_value, f"$.sources[{index}]", issues, context
            )
            if source is not None:
                sources.append(source)
            if source_id is not None:
                source_ids.append(source_id)
        for source_id, count in sorted(Counter(source_ids).items()):
            if count > 1:
                _issue(
                    issues,
                    "$.sources",
                    f"duplicate source ID {source_id!r}",
                    context=context,
                    field="source_id",
                    invalid_value=source_id,
                )

    known_source_ids = set(source_ids)
    species_value = top.get("species", _MISSING)
    species_records = []
    species_ids = []
    if not isinstance(species_value, list):
        _issue(
            issues,
            "$.species",
            "expected an array",
            context=context,
            field="species",
            invalid_value=species_value,
        )
    else:
        if not species_value:
            _issue(
                issues,
                "$.species",
                "dataset must contain at least one exact species prior",
                context=context,
                field="species",
            )
        for index, species_item in enumerate(species_value):
            record, species_id = _species_record(
                species_item,
                f"$.species[{index}]",
                issues,
                context,
                known_source_ids,
            )
            if record is not None:
                species_records.append(record)
            if species_id is not None:
                species_ids.append(species_id)
        for species_id, count in sorted(Counter(species_ids).items()):
            if count > 1:
                _issue(
                    issues,
                    "$.species",
                    f"duplicate exact species record {species_id!r}",
                    context=dict(context, species_id=species_id),
                    field="species_id",
                    invalid_value=species_id,
                )

    if issues:
        raise PublicPriorValidationError(issues)
    assert all(
        value is not None
        for value in (
            schema_version,
            visibility,
            dataset_id,
            dataset_version,
            format_id,
        )
    )
    return PublicPriorDataset(
        schema_version=schema_version,
        visibility=visibility,
        identity=PublicPriorIdentity(dataset_id, dataset_version, format_id),
        patch_version=patch_version,
        display_name=display_name,
        sources=tuple(sources),
        species=tuple(species_records),
        metadata=metadata,
    )
