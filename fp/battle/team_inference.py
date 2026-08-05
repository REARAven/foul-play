"""Per-battle matching and public-evidence filtering of immutable team pools.

This module performs exact opponent team-preview roster matching and
reference-only filtering from public Showdown observations.  It does not
populate hidden battle fields, sample search states, invoke generic set
datasets, or retain complete pool records.  Closed sheets are the default;
open-sheet field exposure is deliberately deferred to a later phase.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Iterable

from fp.battle.helpers import normalize_name
from fp.data import pokedex
from fp.data.team_pools import (
    PoolIdentity,
    RosterKey,
    TeamPoolRegistry,
    TeamRecordId,
    canonical_roster_key,
)


class TeamSheetPolicy(Enum):
    """Explicit team-sheet information policy for one battle."""

    CLOSED = auto()
    OPEN = auto()


class TeamPoolMatchState(Enum):
    """Deterministic lifecycle states for opponent roster matching."""

    UNINITIALIZED = auto()
    NO_POOL = auto()
    INCOMPLETE_PREVIEW = auto()
    MATCHED = auto()
    AMBIGUOUS_MATCH = auto()
    NO_MATCH = auto()
    INVALID_PREVIEW = auto()


class CandidateAccess(Enum):
    """Whether exact candidates may ever be consumed as opponent knowledge."""

    REFERENCE_ONLY = auto()
    OPEN_ELIGIBLE = auto()


class CandidateFilterState(Enum):
    """State of public-evidence filtering, independent of roster matching."""

    NOT_APPLICABLE = auto()
    UNFILTERED = auto()
    CONSISTENT = auto()
    REDUCED = auto()
    EXHAUSTED = auto()
    CONFLICTING_PUBLIC_EVIDENCE = auto()


class PublicObservationSource(Enum):
    """Safe provenance categories for facts visible in Showdown protocol."""

    TEAM_PREVIEW = auto()
    SELECTED_MOVE = auto()
    CLOSING_JAWS_SELECTED_MOVE = auto()
    DIRECT_ITEM_REVEAL = auto()
    ITEM_ACTIVATION = auto()
    ITEM_CONSUMED = auto()
    ITEM_REMOVED = auto()
    DIRECT_ABILITY_REVEAL = auto()
    TRACE_BASE_ABILITY = auto()


@dataclass(frozen=True, order=True)
class TeamPoolCandidateId:
    """Pool-qualified stable identity of a matching team record."""

    pool_identity: PoolIdentity
    team_record_id: TeamRecordId


@dataclass(frozen=True)
class ObservationFilterState:
    """Safe counts and revision for public-observation filtering."""

    revision: int = 0
    applied_observation_count: int = 0
    filter_state: CandidateFilterState = CandidateFilterState.NOT_APPLICABLE
    baseline_candidate_count: int = 0
    active_candidate_count: int = 0


@dataclass(frozen=True)
class OpponentMemberEvidence:
    """Immutable public evidence for one exact preview species/form."""

    species_id: str
    level: int
    selected_move_ids: tuple[str, ...] = ()
    initial_item_id: str | None = None
    base_ability_id: str | None = None
    item_ambiguous: bool = False
    current_ability_changed: bool = False
    conflicting_public_evidence: bool = False
    provenance: tuple[PublicObservationSource, ...] = (
        PublicObservationSource.TEAM_PREVIEW,
    )

    @property
    def applicable_evidence_count(self) -> int:
        return (
            2
            + len(self.selected_move_ids)
            + int(self.initial_item_id is not None and not self.item_ambiguous)
            + int(self.base_ability_id is not None)
        )


@dataclass(frozen=True)
class OpponentObservationLedger:
    """Immutable, species-keyed opponent evidence with no battle objects."""

    members: tuple[OpponentMemberEvidence, ...] = ()
    revision: int = 0

    def member(self, species_id: str) -> OpponentMemberEvidence | None:
        return next(
            (member for member in self.members if member.species_id == species_id),
            None,
        )

    @property
    def has_conflicting_public_evidence(self) -> bool:
        return any(member.conflicting_public_evidence for member in self.members)

    @property
    def applicable_evidence_count(self) -> int:
        return sum(member.applicable_evidence_count for member in self.members)


@dataclass(frozen=True)
class TeamInferenceProvenance:
    """Safe diagnostics containing no submitted set details."""

    source: str
    format_id: str
    preview_member_count: int
    compatible_pool_count: int
    detail: str


@dataclass(frozen=True)
class TeamInferenceSummary:
    """Closed-safe public snapshot for battle and search consumers."""

    policy: TeamSheetPolicy
    candidate_access: CandidateAccess
    match_state: TeamPoolMatchState
    filter_state: CandidateFilterState
    roster_key: RosterKey | None
    selected_pool_identities: tuple[PoolIdentity, ...]
    baseline_candidate_ids: tuple[TeamPoolCandidateId, ...]
    candidate_ids: tuple[TeamPoolCandidateId, ...]
    generic_fallback_eligible: bool
    exact_fields_available: bool
    provenance: TeamInferenceProvenance
    observation_ledger: OpponentObservationLedger
    observation_filter_state: ObservationFilterState


def _public_preview_ledger(preview: tuple[Any, ...]) -> OpponentObservationLedger:
    members: dict[str, OpponentMemberEvidence] = {}
    for pokemon in preview:
        species_id = getattr(pokemon, "name", None)
        level = getattr(pokemon, "level", None)
        if (
            not isinstance(species_id, str)
            or not species_id
            or normalize_name(species_id) != species_id
            or species_id not in pokedex
            or not isinstance(level, int)
            or isinstance(level, bool)
            or level < 1
        ):
            continue
        members.setdefault(
            species_id, OpponentMemberEvidence(species_id=species_id, level=level)
        )
    return OpponentObservationLedger(
        members=tuple(sorted(members.values(), key=lambda member: member.species_id))
    )


class TeamInferenceContext:
    """One battle's safe roster-match state.

    The context never stores a :class:`TeamPoolRegistry`, ``TeamPool``,
    ``TeamRecord``, or ``PokemonRecord``.  Pool records are consulted only by
    explicit matching/filtering calls; ordinary consumers receive immutable
    stable IDs, public observations, and safe summaries.  Even an individual
    closed-sheet candidate does not make exact submitted fields available.
    """

    __slots__ = (
        "_policy",
        "_match_state",
        "_filter_state",
        "_roster_key",
        "_selected_pool_identities",
        "_baseline_candidate_ids",
        "_active_candidate_ids",
        "_generic_fallback_eligible",
        "_provenance",
        "_observation_ledger",
        "_observation_filter_state",
    )

    def __init__(self, policy: TeamSheetPolicy | None = None):
        if policy is None:
            policy = TeamSheetPolicy.CLOSED
        elif not isinstance(policy, TeamSheetPolicy):
            raise TypeError("policy must be a TeamSheetPolicy")

        self._policy = policy
        self._match_state = TeamPoolMatchState.UNINITIALIZED
        self._filter_state = CandidateFilterState.NOT_APPLICABLE
        self._roster_key: RosterKey | None = None
        self._selected_pool_identities: tuple[PoolIdentity, ...] = ()
        self._baseline_candidate_ids: tuple[TeamPoolCandidateId, ...] = ()
        self._active_candidate_ids: tuple[TeamPoolCandidateId, ...] = ()
        self._generic_fallback_eligible = True
        self._provenance = TeamInferenceProvenance(
            source="opponent-team-preview",
            format_id="",
            preview_member_count=0,
            compatible_pool_count=0,
            detail="not initialized",
        )
        self._observation_ledger = OpponentObservationLedger()
        self._observation_filter_state = ObservationFilterState()

    def __deepcopy__(self, memo: dict[int, Any]) -> TeamInferenceContext:
        copied = type(self)(self._policy)
        memo[id(self)] = copied
        copied._match_state = self._match_state
        copied._filter_state = self._filter_state
        copied._roster_key = self._roster_key
        copied._selected_pool_identities = self._selected_pool_identities
        copied._baseline_candidate_ids = self._baseline_candidate_ids
        copied._active_candidate_ids = self._active_candidate_ids
        copied._generic_fallback_eligible = self._generic_fallback_eligible
        copied._provenance = deepcopy(self._provenance, memo)
        copied._observation_ledger = deepcopy(self._observation_ledger, memo)
        copied._observation_filter_state = deepcopy(
            self._observation_filter_state, memo
        )
        return copied

    @property
    def policy(self) -> TeamSheetPolicy:
        return self._policy

    @property
    def candidate_access(self) -> CandidateAccess:
        if self._policy is TeamSheetPolicy.CLOSED:
            return CandidateAccess.REFERENCE_ONLY
        return CandidateAccess.OPEN_ELIGIBLE

    @property
    def match_state(self) -> TeamPoolMatchState:
        return self._match_state

    @property
    def filter_state(self) -> CandidateFilterState:
        return self._filter_state

    @property
    def roster_key(self) -> RosterKey | None:
        return self._roster_key

    @property
    def selected_pool_identities(self) -> tuple[PoolIdentity, ...]:
        return self._selected_pool_identities

    @property
    def matching_pool_identities(self) -> tuple[PoolIdentity, ...]:
        return tuple(
            dict.fromkeys(
                candidate.pool_identity for candidate in self._active_candidate_ids
            )
        )

    @property
    def candidate_ids(self) -> tuple[TeamPoolCandidateId, ...]:
        """Return currently compatible IDs; records remain reference-only."""

        return self._active_candidate_ids

    @property
    def baseline_candidate_ids(self) -> tuple[TeamPoolCandidateId, ...]:
        return self._baseline_candidate_ids

    @property
    def candidate_count(self) -> int:
        return len(self._active_candidate_ids)

    @property
    def baseline_candidate_count(self) -> int:
        return len(self._baseline_candidate_ids)

    @property
    def generic_fallback_eligible(self) -> bool:
        return self._generic_fallback_eligible

    @property
    def exact_fields_available(self) -> bool:
        """Return False until a later, fully tested open-sheet phase."""

        return False

    @property
    def provenance(self) -> TeamInferenceProvenance:
        return self._provenance

    @property
    def observation_ledger(self) -> OpponentObservationLedger:
        return self._observation_ledger

    @property
    def observation_filter_state(self) -> ObservationFilterState:
        return self._observation_filter_state

    def safe_summary(self) -> TeamInferenceSummary:
        """Return the complete closed-safe public view of this context."""

        return TeamInferenceSummary(
            policy=self._policy,
            candidate_access=self.candidate_access,
            match_state=self._match_state,
            filter_state=self._filter_state,
            roster_key=self._roster_key,
            selected_pool_identities=self._selected_pool_identities,
            baseline_candidate_ids=self._baseline_candidate_ids,
            candidate_ids=self._active_candidate_ids,
            generic_fallback_eligible=self._generic_fallback_eligible,
            exact_fields_available=False,
            provenance=self._provenance,
            observation_ledger=self._observation_ledger,
            observation_filter_state=self._observation_filter_state,
        )

    def has_observed_member(self, species_id: str) -> bool:
        return self._observation_ledger.member(species_id) is not None

    def resolve_observed_species(self, *species_ids: str | None) -> str | None:
        """Resolve only an exact ledger ID, without display/base fallbacks."""

        for species_id in species_ids:
            if (
                isinstance(species_id, str)
                and normalize_name(species_id) == species_id
                and self.has_observed_member(species_id)
            ):
                return species_id
        return None

    def _replace_member(self, updated: OpponentMemberEvidence) -> None:
        found = any(
            member.species_id == updated.species_id
            for member in self._observation_ledger.members
        )
        members = tuple(
            sorted(
                tuple(
                    updated if member.species_id == updated.species_id else member
                    for member in self._observation_ledger.members
                )
                + (() if found else (updated,)),
                key=lambda member: member.species_id,
            )
        )
        if members == self._observation_ledger.members:
            return
        self._observation_ledger = OpponentObservationLedger(
            members=members,
            revision=self._observation_ledger.revision + 1,
        )
        if self._baseline_candidate_ids:
            self._filter_state = CandidateFilterState.UNFILTERED
        self._observation_filter_state = ObservationFilterState(
            revision=self._observation_ledger.revision,
            applied_observation_count=0,
            filter_state=self._filter_state,
            baseline_candidate_count=len(self._baseline_candidate_ids),
            active_candidate_count=len(self._active_candidate_ids),
        )

    def record_public_member(self, species_id: str, level: int) -> None:
        """Ensure exact species/form and level exist without a private pool."""

        if (
            not isinstance(species_id, str)
            or not species_id
            or normalize_name(species_id) != species_id
            or species_id not in pokedex
            or isinstance(level, bool)
            or not isinstance(level, int)
            or level < 1
        ):
            return
        existing = self._observation_ledger.member(species_id)
        if existing is not None:
            return
        self._replace_member(
            OpponentMemberEvidence(species_id=species_id, level=level)
        )

    @staticmethod
    def _with_provenance(
        member: OpponentMemberEvidence, source: PublicObservationSource
    ) -> tuple[PublicObservationSource, ...]:
        return tuple(sorted(set(member.provenance + (source,)), key=lambda item: item.value))

    def record_selected_move(
        self,
        species_id: str,
        move_id: str,
        source: PublicObservationSource = PublicObservationSource.SELECTED_MOVE,
    ) -> None:
        member = self._observation_ledger.member(species_id)
        if member is None or not move_id or normalize_name(move_id) != move_id:
            return
        moves = tuple(sorted(set(member.selected_move_ids + (move_id,))))
        self._replace_member(
            replace(
                member,
                selected_move_ids=moves,
                provenance=self._with_provenance(member, source),
            )
        )

    def record_initial_item(
        self,
        species_id: str,
        item_id: str,
        source: PublicObservationSource,
    ) -> None:
        member = self._observation_ledger.member(species_id)
        if (
            member is None
            or member.item_ambiguous
            or not item_id
            or item_id == "none"
            or normalize_name(item_id) != item_id
        ):
            return
        conflict = (
            member.initial_item_id is not None
            and member.initial_item_id != item_id
        )
        self._replace_member(
            replace(
                member,
                initial_item_id=member.initial_item_id or item_id,
                conflicting_public_evidence=(
                    member.conflicting_public_evidence or conflict
                ),
                provenance=self._with_provenance(member, source),
            )
        )

    def mark_item_ambiguous(self, species_id: str) -> None:
        member = self._observation_ledger.member(species_id)
        if member is None:
            return
        self._replace_member(
            replace(member, initial_item_id=None, item_ambiguous=True)
        )

    def record_base_ability(
        self,
        species_id: str,
        ability_id: str,
        source: PublicObservationSource = PublicObservationSource.DIRECT_ABILITY_REVEAL,
    ) -> None:
        member = self._observation_ledger.member(species_id)
        if (
            member is None
            or member.current_ability_changed
            or not ability_id
            or normalize_name(ability_id) != ability_id
        ):
            return
        conflict = (
            member.base_ability_id is not None
            and member.base_ability_id != ability_id
        )
        self._replace_member(
            replace(
                member,
                base_ability_id=member.base_ability_id or ability_id,
                conflicting_public_evidence=(
                    member.conflicting_public_evidence or conflict
                ),
                provenance=self._with_provenance(member, source),
            )
        )

    def mark_current_ability_changed(self, species_id: str) -> None:
        member = self._observation_ledger.member(species_id)
        if member is not None:
            self._replace_member(replace(member, current_ability_changed=True))

    def reset_current_ability_change(self, species_id: str) -> None:
        member = self._observation_ledger.member(species_id)
        if member is not None:
            self._replace_member(replace(member, current_ability_changed=False))

    def _set_result(
        self,
        *,
        state: TeamPoolMatchState,
        format_id: str,
        preview_member_count: int,
        pool_identities: tuple[PoolIdentity, ...],
        roster_key: RosterKey | None = None,
        candidate_ids: tuple[TeamPoolCandidateId, ...] = (),
        observation_ledger: OpponentObservationLedger | None = None,
        detail: str,
    ) -> None:
        self._match_state = state
        self._roster_key = roster_key
        self._selected_pool_identities = pool_identities
        self._baseline_candidate_ids = candidate_ids
        self._active_candidate_ids = candidate_ids
        self._filter_state = (
            CandidateFilterState.UNFILTERED
            if candidate_ids
            else CandidateFilterState.NOT_APPLICABLE
        )
        self._generic_fallback_eligible = True
        self._provenance = TeamInferenceProvenance(
            source="opponent-team-preview",
            format_id=format_id,
            preview_member_count=preview_member_count,
            compatible_pool_count=len(pool_identities),
            detail=detail,
        )
        self._observation_ledger = observation_ledger or OpponentObservationLedger()
        self._observation_filter_state = ObservationFilterState(
            revision=self._observation_ledger.revision,
            filter_state=self._filter_state,
            baseline_candidate_count=len(candidate_ids),
            active_candidate_count=len(candidate_ids),
        )

    def match_preview(
        self,
        preview_pokemon: Iterable[Any],
        registry: TeamPoolRegistry | None,
        format_id: str,
    ) -> None:
        """Match an exact six-member opponent preview without exposing sets."""

        preview = tuple(preview_pokemon)
        observation_ledger = _public_preview_ledger(preview)
        if registry is not None and not isinstance(registry, TeamPoolRegistry):
            raise TypeError("registry must be a TeamPoolRegistry or None")

        if registry is None:
            self._set_result(
                state=TeamPoolMatchState.NO_POOL,
                format_id=format_id,
                preview_member_count=len(preview),
                pool_identities=(),
                observation_ledger=observation_ledger,
                detail="no registry supplied",
            )
            return

        compatible_pools = tuple(
            pool for pool in registry if pool.identity.format_id == format_id
        )
        pool_identities = tuple(pool.identity for pool in compatible_pools)
        if not compatible_pools:
            self._set_result(
                state=TeamPoolMatchState.NO_POOL,
                format_id=format_id,
                preview_member_count=len(preview),
                pool_identities=(),
                observation_ledger=observation_ledger,
                detail="registry has no pool for the exact battle format",
            )
            return

        if len(preview) < 6:
            self._set_result(
                state=TeamPoolMatchState.INCOMPLETE_PREVIEW,
                format_id=format_id,
                preview_member_count=len(preview),
                pool_identities=pool_identities,
                observation_ledger=observation_ledger,
                detail="preview has fewer than six members",
            )
            return
        if len(preview) > 6:
            self._set_result(
                state=TeamPoolMatchState.INVALID_PREVIEW,
                format_id=format_id,
                preview_member_count=len(preview),
                pool_identities=pool_identities,
                observation_ledger=observation_ledger,
                detail="preview has more than six members",
            )
            return

        species_ids = []
        base_species_ids = []
        for pokemon in preview:
            species_id = getattr(pokemon, "name", None)
            level = getattr(pokemon, "level", None)
            if (
                getattr(pokemon, "unknown_forme", False)
                or not isinstance(species_id, str)
                or not species_id
                or normalize_name(species_id) != species_id
                or species_id not in pokedex
                or not isinstance(level, int)
                or isinstance(level, bool)
                or level < 1
            ):
                self._set_result(
                    state=TeamPoolMatchState.INVALID_PREVIEW,
                    format_id=format_id,
                    preview_member_count=len(preview),
                    pool_identities=pool_identities,
                    observation_ledger=observation_ledger,
                    detail="preview contains an unknown or unresolved exact form",
                )
                return

            species_data = pokedex[species_id]
            raw_base_species = species_data.get("baseSpecies")
            if raw_base_species is None:
                base_species_id = species_id
            elif isinstance(raw_base_species, str):
                base_species_id = normalize_name(raw_base_species)
                if base_species_id not in pokedex:
                    self._set_result(
                        state=TeamPoolMatchState.INVALID_PREVIEW,
                        format_id=format_id,
                        preview_member_count=len(preview),
                        pool_identities=pool_identities,
                        observation_ledger=observation_ledger,
                        detail="preview base species cannot be resolved authoritatively",
                    )
                    return
            else:
                self._set_result(
                    state=TeamPoolMatchState.INVALID_PREVIEW,
                    format_id=format_id,
                    preview_member_count=len(preview),
                    pool_identities=pool_identities,
                    observation_ledger=observation_ledger,
                    detail="preview base-species metadata is malformed",
                )
                return

            species_ids.append(species_id)
            base_species_ids.append(base_species_id)

        if len(set(base_species_ids)) != 6:
            self._set_result(
                state=TeamPoolMatchState.INVALID_PREVIEW,
                format_id=format_id,
                preview_member_count=len(preview),
                pool_identities=pool_identities,
                observation_ledger=observation_ledger,
                detail="preview violates Species Clause by base-species identity",
            )
            return

        roster_key = canonical_roster_key(species_ids)
        candidates = tuple(
            sorted(
                TeamPoolCandidateId(pool.identity, team_record_id)
                for pool in compatible_pools
                for team_record_id in pool.matching_team_ids(roster_key)
            )
        )
        if not candidates:
            state = TeamPoolMatchState.NO_MATCH
            detail = "no exact roster match"
        elif len(candidates) == 1:
            state = TeamPoolMatchState.MATCHED
            detail = "one closed-safe candidate ID matched"
        else:
            state = TeamPoolMatchState.AMBIGUOUS_MATCH
            detail = "multiple closed-safe candidate IDs matched"

        self._set_result(
            state=state,
            format_id=format_id,
            preview_member_count=len(preview),
            pool_identities=pool_identities,
            roster_key=roster_key,
            candidate_ids=candidates,
            observation_ledger=observation_ledger,
            detail=detail,
        )


def _candidate_is_compatible(
    candidate: TeamPoolCandidateId,
    ledger: OpponentObservationLedger,
    registry: TeamPoolRegistry,
) -> bool:
    """Compare public evidence only; never return or retain a pool record."""

    pool = registry.get(candidate.pool_identity)
    if pool is None:
        return False
    team = pool.get_team(candidate.team_record_id)
    if team is None:
        return False

    members_by_species = {member.species_id: member for member in team.pokemon}
    if len(members_by_species) != len(team.pokemon):
        return False

    for evidence in ledger.members:
        member = members_by_species.get(evidence.species_id)
        if member is None or member.level != evidence.level:
            return False
        if not set(evidence.selected_move_ids).issubset(member.move_ids):
            return False
        if not evidence.conflicting_public_evidence:
            if (
                evidence.initial_item_id is not None
                and not evidence.item_ambiguous
                and member.item_id != evidence.initial_item_id
            ):
                return False
            if (
                evidence.base_ability_id is not None
                and member.base_ability_id != evidence.base_ability_id
            ):
                return False
    return True


def filter_team_candidates(
    context: TeamInferenceContext,
    registry: TeamPoolRegistry | None,
) -> TeamInferenceContext:
    """Return an independently copied context filtered from its baseline IDs.

    Registry and record objects are used only during this call and are never
    retained.  Exact records remain reference-only under CLOSED policy, even
    when filtering leaves one candidate.
    """

    if not isinstance(context, TeamInferenceContext):
        raise TypeError("context must be a TeamInferenceContext")
    if registry is not None and not isinstance(registry, TeamPoolRegistry):
        raise TypeError("registry must be a TeamPoolRegistry or None")

    filtered = deepcopy(context)
    baseline = filtered._baseline_candidate_ids
    if (
        registry is None
        or not baseline
        or filtered._match_state
        not in (TeamPoolMatchState.MATCHED, TeamPoolMatchState.AMBIGUOUS_MATCH)
    ):
        filtered._filter_state = CandidateFilterState.NOT_APPLICABLE
        filtered._observation_filter_state = ObservationFilterState(
            revision=filtered._observation_ledger.revision,
            filter_state=filtered._filter_state,
            baseline_candidate_count=len(baseline),
            active_candidate_count=len(filtered._active_candidate_ids),
        )
        return filtered

    active = tuple(
        candidate
        for candidate in baseline
        if _candidate_is_compatible(candidate, filtered._observation_ledger, registry)
    )
    filtered._active_candidate_ids = active
    if filtered._observation_ledger.has_conflicting_public_evidence:
        state = CandidateFilterState.CONFLICTING_PUBLIC_EVIDENCE
    elif len(active) == len(baseline):
        state = CandidateFilterState.CONSISTENT
    elif active:
        state = CandidateFilterState.REDUCED
    else:
        state = CandidateFilterState.EXHAUSTED

    filtered._filter_state = state
    filtered._observation_filter_state = ObservationFilterState(
        revision=filtered._observation_ledger.revision,
        applied_observation_count=(
            filtered._observation_ledger.applicable_evidence_count
        ),
        filter_state=state,
        baseline_candidate_count=len(baseline),
        active_candidate_count=len(active),
    )
    return filtered
