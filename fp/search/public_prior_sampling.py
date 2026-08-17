"""Public-only opponent variant selection and copied-state population."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol

from fp import constants
from fp.battle.helpers import normalize_name, possible_hidden_power_types
from fp.battle.public_prior_context import PublicPriorSearchContext
from fp.data.public_priors import (
    NO_ITEM_ID,
    PublicPriorIdentity,
    PublicSetVariant,
)


class PublicEvidence(Protocol):
    """The safe public subset consumed from the observation ledger."""

    species_id: str
    level: int
    selected_move_ids: tuple[str, ...]
    initial_item_id: str | None
    base_ability_id: str | None
    item_ambiguous: bool
    current_ability_changed: bool
    conflicting_public_evidence: bool


class PublicPriorSelectionStatus(Enum):
    SELECTED = "selected"
    CONTEXT_FORMAT_MISMATCH = "context_format_mismatch"
    NO_COMPATIBLE_VARIANT = "no_compatible_variant"


@dataclass(frozen=True)
class PublicPriorSelectionResult:
    status: PublicPriorSelectionStatus
    dataset_identity: PublicPriorIdentity | None = None
    variant: PublicSetVariant | None = None


_HIDDEN_POWER_TYPES = frozenset(possible_hidden_power_types())
_HIDDEN_POWER_POWER_SUFFIXES = ("60", "70")


def _hidden_power_type(move_id: str) -> str | None:
    """Return ``""`` for generic Hidden Power, its type, or ``None``."""

    normalized = normalize_name(move_id)
    if normalized == constants.HIDDEN_POWER:
        return ""
    if not normalized.startswith(constants.HIDDEN_POWER):
        return None

    type_and_power = normalized[len(constants.HIDDEN_POWER) :]
    for power_suffix in _HIDDEN_POWER_POWER_SUFFIXES:
        if type_and_power.endswith(power_suffix):
            type_and_power = type_and_power[: -len(power_suffix)]
            break
    return type_and_power if type_and_power in _HIDDEN_POWER_TYPES else None


def observed_move_is_compatible_with_candidate_move(
    observed_move_id: str, candidate_move_id: str
) -> bool:
    """Compare one observed move with one complete-candidate move.

    Ordinary moves retain exact normalized-ID semantics. Generic Hidden Power
    proves only family membership, while a typed observation must match the
    candidate's type. The candidate ID is never rewritten.
    """

    observed = normalize_name(observed_move_id)
    candidate = normalize_name(candidate_move_id)
    if observed == candidate:
        return True

    observed_type = _hidden_power_type(observed)
    candidate_type = _hidden_power_type(candidate)
    if observed_type is None or candidate_type is None:
        return False
    if observed_type == "":
        return True
    return candidate_type != "" and observed_type == candidate_type


def _observed_moves_are_compatible_with_candidate_moves(
    observed_move_ids: Iterable[str], candidate_move_ids: Iterable[str]
) -> bool:
    candidates = tuple(candidate_move_ids)
    return all(
        any(
            observed_move_is_compatible_with_candidate_move(observed, candidate)
            for candidate in candidates
        )
        for observed in observed_move_ids
    )


def candidate_original_item_is_compatible(
    candidate_item_id: str, evidence: PublicEvidence | None
) -> bool:
    """Compare a candidate's authored item with confident historical evidence.

    Current held-item state is intentionally absent from this comparison.  A
    consumed, removed, transferred, or replaced item remains the candidate's
    original item, while ambiguous acquisition without an earlier reveal does
    not fabricate one.
    """

    if evidence is None or evidence.initial_item_id is None:
        return True
    return candidate_item_id == evidence.initial_item_id


def public_variant_is_compatible(
    variant: PublicSetVariant,
    species_id: str,
    level: int,
    evidence: PublicEvidence | None,
) -> bool:
    """Compare one coherent variant with public evidence only."""

    if variant.level != level:
        return False
    if evidence is None:
        return True
    if evidence.species_id != species_id or evidence.level != level:
        return False
    if evidence.conflicting_public_evidence:
        return False
    if not _observed_moves_are_compatible_with_candidate_moves(
        evidence.selected_move_ids, variant.move_ids
    ):
        return False
    if not candidate_original_item_is_compatible(variant.item_id, evidence):
        return False
    if (
        evidence.base_ability_id is not None
        and variant.base_ability_id != evidence.base_ability_id
    ):
        return False
    return True


def choose_weighted_public_variant(
    variants: Iterable[PublicSetVariant], rng: Any = None
) -> PublicSetVariant | None:
    """Select from sorted compatible variants without mutating authored weights."""

    ordered = tuple(sorted(variants, key=lambda variant: variant.variant_id))
    if not ordered:
        return None
    rng = random if rng is None else rng
    maximum = max(variant.weight for variant in ordered)
    scaled_weights = tuple(variant.weight / maximum for variant in ordered)
    target = rng.random() * sum(scaled_weights)
    cumulative = 0.0
    for variant, weight in zip(ordered, scaled_weights):
        cumulative += weight
        if target < cumulative:
            return variant
    return ordered[-1]


def _compatible_family_candidates(
    context: PublicPriorSearchContext,
    identities: tuple[PublicPriorIdentity, ...],
    *,
    battle_format: str,
    species_id: str,
    level: int,
    evidence: PublicEvidence | None,
) -> tuple[tuple[PublicPriorIdentity, PublicSetVariant], ...]:
    """Return compatible variants from one dataset family in layer order.

    Identities sharing a ``dataset_id`` are version layers of one public prior
    family. A variant ID from an earlier-selected layer shadows the same ID in
    later layers, while unique variant IDs remain additive. This lets a small
    newer public-prior delta extend an immutable older release without making
    unrelated dataset families lose their existing precedence semantics.
    """

    candidates = []
    shadowed_variant_ids = set()
    for identity in identities:
        if identity.format_id != battle_format:
            continue
        dataset = context.registry.get(identity)
        if dataset is None or dataset.identity.format_id != battle_format:
            continue
        species = dataset.get_species(species_id)
        if species is None:
            continue
        for variant in species.variants:
            if variant.variant_id in shadowed_variant_ids:
                continue
            shadowed_variant_ids.add(variant.variant_id)
            if public_variant_is_compatible(variant, species_id, level, evidence):
                candidates.append((identity, variant))
    return tuple(candidates)


def select_public_prior_variant(
    context: PublicPriorSearchContext,
    *,
    battle_format: str,
    species_id: str,
    level: int,
    evidence: PublicEvidence | None,
    rng: Any = None,
) -> PublicPriorSelectionResult:
    """Apply dataset-family precedence and layer compatible family variants."""

    if context.format_id != battle_format:
        return PublicPriorSelectionResult(
            PublicPriorSelectionStatus.CONTEXT_FORMAT_MISMATCH
        )

    seen_dataset_ids = set()
    for identity in context.selected_identities:
        if identity.format_id != battle_format:
            continue
        if identity.dataset_id in seen_dataset_ids:
            continue
        seen_dataset_ids.add(identity.dataset_id)
        family_identities = tuple(
            candidate_identity
            for candidate_identity in context.selected_identities
            if candidate_identity.dataset_id == identity.dataset_id
        )
        candidates = _compatible_family_candidates(
            context,
            family_identities,
            battle_format=battle_format,
            species_id=species_id,
            level=level,
            evidence=evidence,
        )
        if not candidates:
            continue
        selected_variant = choose_weighted_public_variant(
            (variant for _, variant in candidates), rng=rng
        )
        if selected_variant is None:
            continue
        selected_identity = next(
            candidate_identity
            for candidate_identity, variant in candidates
            if variant is selected_variant
        )
        return PublicPriorSelectionResult(
            status=PublicPriorSelectionStatus.SELECTED,
            dataset_identity=selected_identity,
            variant=selected_variant,
        )

    return PublicPriorSelectionResult(PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT)


def _populate_item(
    pokemon: Any, variant: PublicSetVariant, evidence: PublicEvidence | None
) -> None:
    item_ambiguous = evidence is not None and evidence.item_ambiguous
    item_was_removed = (
        pokemon.removed_item is not None or pokemon.knocked_off or pokemon.item is None
    )
    if item_ambiguous or item_was_removed:
        return
    if pokemon.item == constants.UNKNOWN_ITEM or pokemon.item_inferred:
        pokemon.item = None if variant.item_id == NO_ITEM_ID else variant.item_id
        pokemon.item_inferred = False


def _populate_ability(
    pokemon: Any, variant: PublicSetVariant, evidence: PublicEvidence | None
) -> None:
    public_base = (
        evidence.base_ability_id
        if evidence is not None and evidence.base_ability_id is not None
        else variant.base_ability_id
    )
    current_changed = (evidence is not None and evidence.current_ability_changed) or (
        pokemon.original_ability is not None
        and pokemon.ability != pokemon.original_ability
    )
    if current_changed:
        if pokemon.original_ability is None:
            pokemon.original_ability = public_base
        return
    pokemon.original_ability = public_base
    pokemon.ability = public_base


def populate_pokemon_from_public_variant(
    pokemon: Any,
    variant: PublicSetVariant,
    evidence: PublicEvidence | None,
) -> bool:
    """Populate one copied Pokémon while preserving all public runtime state.

    Returns ``False`` without mutation when existing moves would require a
    five-move state. A transformed runtime state is intentionally left intact
    and counts as handled so generic sampling cannot overwrite it.
    """

    if constants.TRANSFORM in pokemon.volatile_statuses:
        return True

    existing_moves = tuple(pokemon.moves)
    if len(existing_moves) > 4:
        return False

    resolved_moves = []
    resolved_candidate_ids = set()
    for move in existing_moves:
        compatible_candidate_ids = tuple(
            candidate_move_id
            for candidate_move_id in variant.move_ids
            if observed_move_is_compatible_with_candidate_move(
                move.name, candidate_move_id
            )
        )
        if (
            len(compatible_candidate_ids) != 1
            or compatible_candidate_ids[0] in resolved_candidate_ids
        ):
            return False
        candidate_move_id = compatible_candidate_ids[0]
        resolved_candidate_ids.add(candidate_move_id)
        if move.name == candidate_move_id:
            resolved_moves.append(move)
        else:
            resolved_move = copy.copy(move)
            resolved_move.name = candidate_move_id
            resolved_moves.append(resolved_move)

    missing_move_ids = tuple(
        move_id for move_id in variant.move_ids if move_id not in resolved_candidate_ids
    )
    if len(resolved_moves) + len(missing_move_ids) > 4:
        return False
    _populate_item(pokemon, variant, evidence)
    _populate_ability(pokemon, variant, evidence)

    pokemon.level = variant.level
    pokemon.set_spread(
        variant.nature_id,
        variant.evs.as_tuple(),
        ivs=variant.ivs.as_tuple(),
    )

    pokemon.moves = resolved_moves
    for move_id in missing_move_ids:
        pokemon.add_move(move_id)
    return len(pokemon.moves) == 4
