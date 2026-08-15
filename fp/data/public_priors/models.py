"""Immutable schema-v1 public opponent-prior records and indexes.

These public archetypes are intentionally unrelated to the private complete-team
reference-pool types.  They contain no roster identity, trainer identity, private
candidate ID, battle object, or runtime state.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from fp.battle.helpers import normalize_name

from .errors import PublicPriorRegistryError


SCHEMA_VERSION = 1
PUBLIC_VISIBILITY = "public"
NO_ITEM_ID = "none"
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


class PublicSourceKind(str, Enum):
    """The deliberately narrow source kinds authorized by schema v1."""

    PUBLIC_FORMAT = "public_format"
    PUBLIC_REPLAY = "public_replay"
    PUBLIC_MANUAL = "public_manual"
    SYNTHETIC_TEST = "synthetic_test"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or normalize_name(value) != value:
        raise ValueError(f"{label} must be a nonempty canonical normalized ID")
    return value


@dataclass(frozen=True, order=True)
class PublicPriorIdentity:
    """Stable versioned, format-scoped identity of one public dataset."""

    dataset_id: str
    dataset_version: str
    format_id: str

    def __post_init__(self) -> None:
        _canonical_identifier(self.dataset_id, "dataset ID")
        if not isinstance(self.dataset_version, str) or not self.dataset_version:
            raise ValueError("dataset version must be a nonempty string")
        _canonical_identifier(self.format_id, "format ID")


@dataclass(frozen=True, order=True)
class PublicVariantReference:
    """Public-only dataset-qualified reference to one coherent set variant."""

    dataset_identity: PublicPriorIdentity
    species_id: str
    variant_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_identity, PublicPriorIdentity):
            raise TypeError("dataset_identity must be a PublicPriorIdentity")
        _canonical_identifier(self.species_id, "species ID")
        _canonical_identifier(self.variant_id, "variant ID")


@dataclass(frozen=True)
class PublicStatValues:
    """Immutable values for the six standard stats."""

    hp: int
    atk: int
    defense: int
    spa: int
    spd: int
    spe: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.hp, self.atk, self.defense, self.spa, self.spd, self.spe)


@dataclass(frozen=True)
class PublicSource:
    """One explicit public authorization/evidence declaration."""

    source_id: str
    kind: PublicSourceKind
    public_date: str | None
    public_version: str | None
    description: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _canonical_identifier(self.source_id, "source ID")
        if not isinstance(self.kind, PublicSourceKind):
            raise TypeError("kind must be a PublicSourceKind")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True)
class PublicSetVariant:
    """One complete coherent public set, never a cross-product of marginals."""

    variant_id: str
    weight: float
    item_id: str
    base_ability_id: str
    move_ids: tuple[str, str, str, str]
    nature_id: str
    evs: PublicStatValues
    ivs: PublicStatValues
    level: int
    source_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _canonical_identifier(self.variant_id, "variant ID")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("weight must be a positive finite number")
        object.__setattr__(self, "weight", float(self.weight))
        _canonical_identifier(self.item_id, "item ID")
        _canonical_identifier(self.base_ability_id, "base ability ID")
        moves = tuple(self.move_ids)
        if len(moves) != 4 or len(set(moves)) != 4:
            raise ValueError("move_ids must contain exactly four distinct IDs")
        for move_id in moves:
            _canonical_identifier(move_id, "move ID")
        _canonical_identifier(self.nature_id, "nature ID")
        if not isinstance(self.evs, PublicStatValues):
            raise TypeError("evs must be PublicStatValues")
        if not isinstance(self.ivs, PublicStatValues):
            raise TypeError("ivs must be PublicStatValues")
        if isinstance(self.level, bool) or not isinstance(self.level, int):
            raise TypeError("level must be an integer")
        sources = tuple(self.source_ids)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("source_ids must be nonempty and unique")
        for source_id in sources:
            _canonical_identifier(source_id, "source ID")
        object.__setattr__(self, "move_ids", moves)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True)
class SpeciesPrior:
    """Public variants for one exact species/form ID."""

    species_id: str
    base_species_id: str | None
    variants: tuple[PublicSetVariant, ...]
    metadata: Mapping[str, Any]
    _variant_lookup: Mapping[str, PublicSetVariant] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _canonical_identifier(self.species_id, "species ID")
        if self.base_species_id is not None:
            _canonical_identifier(self.base_species_id, "base species ID")
        variants = tuple(self.variants)
        if not variants:
            raise ValueError("a species prior requires at least one variant")
        if not all(isinstance(variant, PublicSetVariant) for variant in variants):
            raise TypeError("variants must contain only PublicSetVariant values")
        variants = tuple(sorted(variants, key=lambda variant: variant.variant_id))
        lookup = {variant.variant_id: variant for variant in variants}
        if len(lookup) != len(variants):
            raise ValueError("duplicate public variant ID")
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(self, "_variant_lookup", MappingProxyType(lookup))

    @property
    def variant_lookup(self) -> Mapping[str, PublicSetVariant]:
        return self._variant_lookup

    def get_variant(self, variant_id: str) -> PublicSetVariant | None:
        return self._variant_lookup.get(variant_id)


@dataclass(frozen=True, order=True)
class NormalizedVariantProbability:
    """Immutable derived probability for one dataset-qualified variant."""

    reference: PublicVariantReference
    probability: float


@dataclass(frozen=True)
class PublicPriorDataset:
    """One immutable public dataset with deterministic exact-form indexes."""

    schema_version: int
    visibility: str
    identity: PublicPriorIdentity
    patch_version: str | None
    display_name: str | None
    sources: tuple[PublicSource, ...]
    species: tuple[SpeciesPrior, ...]
    metadata: Mapping[str, Any]
    _source_lookup: Mapping[str, PublicSource] = field(init=False, repr=False)
    _species_lookup: Mapping[str, SpeciesPrior] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.visibility != PUBLIC_VISIBILITY:
            raise ValueError(f"visibility must be {PUBLIC_VISIBILITY!r}")
        if not isinstance(self.identity, PublicPriorIdentity):
            raise TypeError("identity must be a PublicPriorIdentity")
        sources = tuple(self.sources)
        species = tuple(self.species)
        if not sources or not all(
            isinstance(source, PublicSource) for source in sources
        ):
            raise TypeError("sources must contain one or more PublicSource values")
        if not species or not all(
            isinstance(record, SpeciesPrior) for record in species
        ):
            raise TypeError("species must contain one or more SpeciesPrior values")
        sources = tuple(sorted(sources, key=lambda source: source.source_id))
        species = tuple(sorted(species, key=lambda record: record.species_id))
        source_lookup = {source.source_id: source for source in sources}
        species_lookup = {record.species_id: record for record in species}
        if len(source_lookup) != len(sources):
            raise ValueError("duplicate public source ID")
        if len(species_lookup) != len(species):
            raise ValueError("duplicate exact species prior")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(self, "_source_lookup", MappingProxyType(source_lookup))
        object.__setattr__(self, "_species_lookup", MappingProxyType(species_lookup))

    @property
    def source_lookup(self) -> Mapping[str, PublicSource]:
        return self._source_lookup

    @property
    def species_lookup(self) -> Mapping[str, SpeciesPrior]:
        return self._species_lookup

    def get_species(self, species_id: str) -> SpeciesPrior | None:
        """Return only an exact-form match; never use display/base prefixes."""

        return self._species_lookup.get(species_id)

    def variant_references(self, species_id: str) -> tuple[PublicVariantReference, ...]:
        record = self.get_species(species_id)
        if record is None:
            return ()
        return tuple(
            PublicVariantReference(self.identity, species_id, variant.variant_id)
            for variant in record.variants
        )

    def normalized_probabilities(
        self, species_id: str
    ) -> tuple[NormalizedVariantProbability, ...]:
        """Derive stable probabilities without modifying authored weights.

        Scaling by the largest weight avoids overflow when multiple very large,
        individually finite weights are present.
        """

        record = self.get_species(species_id)
        if record is None:
            return ()
        maximum = max(variant.weight for variant in record.variants)
        scaled = tuple(variant.weight / maximum for variant in record.variants)
        denominator = sum(scaled)
        return tuple(
            NormalizedVariantProbability(
                reference=PublicVariantReference(
                    self.identity, species_id, variant.variant_id
                ),
                probability=scaled_weight / denominator,
            )
            for variant, scaled_weight in zip(record.variants, scaled)
        )


@dataclass(frozen=True)
class PublicPriorRegistry:
    """Immutable collection of public datasets with no merge/precedence rules."""

    datasets: tuple[PublicPriorDataset, ...] = ()
    _dataset_lookup: Mapping[PublicPriorIdentity, PublicPriorDataset] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if not all(isinstance(dataset, PublicPriorDataset) for dataset in datasets):
            raise TypeError("datasets must contain only PublicPriorDataset values")
        datasets = tuple(sorted(datasets, key=lambda dataset: dataset.identity))
        lookup: dict[PublicPriorIdentity, PublicPriorDataset] = {}
        for dataset in datasets:
            if dataset.identity in lookup:
                raise PublicPriorRegistryError(
                    f"duplicate public-prior identity: {dataset.identity!r}"
                )
            lookup[dataset.identity] = dataset
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "_dataset_lookup", MappingProxyType(lookup))

    def __iter__(self) -> Iterator[PublicPriorDataset]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    @property
    def dataset_lookup(self) -> Mapping[PublicPriorIdentity, PublicPriorDataset]:
        return self._dataset_lookup

    def get(self, identity: PublicPriorIdentity) -> PublicPriorDataset | None:
        return self._dataset_lookup.get(identity)

    def get_species(
        self, identity: PublicPriorIdentity, species_id: str
    ) -> SpeciesPrior | None:
        dataset = self.get(identity)
        return None if dataset is None else dataset.get_species(species_id)

    def references_for_species(
        self,
        species_id: str,
        dataset_identities: tuple[PublicPriorIdentity, ...],
    ) -> tuple[PublicVariantReference, ...]:
        """Return qualified references across explicit datasets without merging."""

        if not isinstance(dataset_identities, tuple):
            raise TypeError("dataset_identities must be an explicit tuple")
        if not all(
            isinstance(identity, PublicPriorIdentity) for identity in dataset_identities
        ):
            raise TypeError("dataset identities must be PublicPriorIdentity values")
        references = {
            reference
            for identity in dataset_identities
            for dataset in (self.get(identity),)
            if dataset is not None
            for reference in dataset.variant_references(species_id)
        }
        return tuple(sorted(references))
