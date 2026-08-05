"""Public-only opponent variant selection and copied-state population."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol

from fp import constants
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
    if not set(evidence.selected_move_ids).issubset(variant.move_ids):
        return False
    if (
        evidence.initial_item_id is not None
        and not evidence.item_ambiguous
        and variant.item_id != evidence.initial_item_id
    ):
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


def select_public_prior_variant(
    context: PublicPriorSearchContext,
    *,
    battle_format: str,
    species_id: str,
    level: int,
    evidence: PublicEvidence | None,
    rng: Any = None,
) -> PublicPriorSelectionResult:
    """Apply configured dataset precedence and return one public variant."""

    if context.format_id != battle_format:
        return PublicPriorSelectionResult(
            PublicPriorSelectionStatus.CONTEXT_FORMAT_MISMATCH
        )

    for identity in context.selected_identities:
        if identity.format_id != battle_format:
            continue
        dataset = context.registry.get(identity)
        if dataset is None or dataset.identity.format_id != battle_format:
            continue
        species = dataset.get_species(species_id)
        if species is None:
            continue
        compatible = tuple(
            variant
            for variant in species.variants
            if public_variant_is_compatible(
                variant, species_id, level, evidence
            )
        )
        if not compatible:
            continue
        return PublicPriorSelectionResult(
            status=PublicPriorSelectionStatus.SELECTED,
            dataset_identity=identity,
            variant=choose_weighted_public_variant(compatible, rng=rng),
        )

    return PublicPriorSelectionResult(
        PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT
    )


def _populate_item(
    pokemon: Any, variant: PublicSetVariant, evidence: PublicEvidence | None
) -> None:
    item_ambiguous = evidence is not None and evidence.item_ambiguous
    item_was_removed = (
        pokemon.removed_item is not None
        or pokemon.knocked_off
        or pokemon.item is None
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
    current_changed = (
        evidence is not None and evidence.current_ability_changed
    ) or (
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
    five-move state.  A transformed runtime state is intentionally left intact
    and counts as handled so generic sampling cannot overwrite it.
    """

    if constants.TRANSFORM in pokemon.volatile_statuses:
        return True

    existing_moves = tuple(pokemon.moves)
    existing_move_ids = tuple(move.name for move in existing_moves)
    if len(existing_moves) > 4 or not set(existing_move_ids).issubset(
        variant.move_ids
    ):
        return False
    missing_move_ids = tuple(
        move_id for move_id in variant.move_ids if move_id not in existing_move_ids
    )
    if len(existing_moves) + len(missing_move_ids) > 4:
        return False
    _populate_item(pokemon, variant, evidence)
    _populate_ability(pokemon, variant, evidence)

    pokemon.level = variant.level
    pokemon.set_spread(
        variant.nature_id,
        variant.evs.as_tuple(),
        ivs=variant.ivs.as_tuple(),
    )

    pokemon.moves = list(existing_moves)
    for move_id in missing_move_ids:
        pokemon.add_move(move_id)
    return len(pokemon.moves) == 4
