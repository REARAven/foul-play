import builtins
import copy
import socket
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

from fp import constants
from fp.battle.protocol import (
    activate,
    fieldstart,
    process_battle_updates,
    remove_item,
    unlikely_to_have_choice_item,
)
from fp.battle.public_prior_context import (
    PublicPriorFallback,
    PublicPriorSearchContext,
    SelectedDatasetStatus,
)
from fp.battle.state import Battle, LastUsedMove, Pokemon
from fp.battle.team_inference import (
    CandidateFilterState,
    OpponentMemberEvidence,
    PublicObservationSource,
)
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import (
    PublicPriorDataset,
    PublicPriorIdentity,
    PublicPriorRegistry,
    PublicSetVariant,
    PublicSource,
    PublicSourceKind,
    PublicStatValues,
    SpeciesPrior,
)
from fp.format_spec import FormatSpec
from fp.modes.standard_battle import StandardBattleMode
from fp.search.poke_engine_helpers import (
    battle_to_poke_engine_state,
    pokemon_to_poke_engine_pkmn,
)
from fp.search.public_prior_sampling import (
    PublicPriorSelectionStatus,
    candidate_original_item_is_compatible,
    choose_weighted_public_variant,
    observed_move_is_compatible_with_candidate_move,
    populate_pokemon_from_public_variant,
    public_variant_is_compatible,
    select_public_prior_variant,
)
from fp.search.standard_battles import prepare_battles


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FORMAT = FoulPlayConfig.pokemon_format


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"


def tearDownModule():
    FoulPlayConfig.pokemon_format = ORIGINAL_FORMAT


def _stats(hp=31, atk=31, defense=31, spa=31, spd=31, spe=31):
    return PublicStatValues(hp, atk, defense, spa, spd, spe)


def _variant(
    variant_id="standard",
    *,
    weight=1,
    item="lightball",
    ability="static",
    moves=("thunderbolt", "voltswitch", "surf", "protect"),
    nature="timid",
    evs=None,
    ivs=None,
    level=50,
):
    return PublicSetVariant(
        variant_id=variant_id,
        weight=weight,
        item_id=item,
        base_ability_id=ability,
        move_ids=tuple(moves),
        nature_id=nature,
        evs=evs or _stats(hp=0, atk=0, defense=0, spa=252, spd=4, spe=252),
        ivs=ivs or _stats(),
        level=level,
        source_ids=("synthetic",),
        metadata={"synthetic": True},
    )


def _dataset(
    dataset_id="syntheticprior",
    *,
    version="1",
    format_id="gen9tugs",
    species_id="pikachu",
    variants=None,
):
    identity = PublicPriorIdentity(dataset_id, version, format_id)
    return PublicPriorDataset(
        schema_version=1,
        visibility="public",
        identity=identity,
        patch_version="synthetic",
        display_name="Synthetic public prior",
        sources=(
            PublicSource(
                source_id="synthetic",
                kind=PublicSourceKind.SYNTHETIC_TEST,
                public_date=None,
                public_version=None,
                description="Synthetic test data only",
                metadata={},
            ),
        ),
        species=(
            SpeciesPrior(
                species_id=species_id,
                base_species_id=None,
                variants=tuple(variants or (_variant(),)),
                metadata={},
            ),
        ),
        metadata={"synthetic": True},
    )


def _context(
    *datasets, selected=None, fallback=PublicPriorFallback.NONE, format_id="gen9tugs"
):
    datasets = tuple(datasets or (_dataset(),))
    identities = tuple(selected or tuple(dataset.identity for dataset in datasets))
    return PublicPriorSearchContext(
        registry=PublicPriorRegistry(datasets),
        selected_identities=identities,
        fallback_policy=fallback,
        format_id=format_id,
    )


def _evidence(
    *,
    species="pikachu",
    level=50,
    moves=(),
    item=None,
    ability=None,
    ambiguous=False,
    changed=False,
    conflict=False,
):
    return OpponentMemberEvidence(
        species_id=species,
        level=level,
        selected_move_ids=tuple(moves),
        initial_item_id=item,
        base_ability_id=ability,
        item_ambiguous=ambiguous,
        current_ability_changed=changed,
        conflicting_public_evidence=conflict,
    )


class _Mode:
    smogon_sets = None

    def __deepcopy__(self, memo):
        return self

    def sample_mega_evolution(self, battle, index, smogon_sets):
        return None

    def check_zoroark_from_move(
        self, battle, side, pokemon, move_name, split_msg, zoroark_from_reserves
    ):
        return pokemon

    def assume_spread_for_speed_check(self, battle, battle_copy):
        return None


def _battle(context=None, *, format_id="gen9tugs", species="pikachu", level=50):
    battle = Battle("synthetic", public_prior_context=context)
    battle.pokemon_format = format_id
    battle.generation = FormatSpec.from_format_string(format_id).generation
    battle.mode = _Mode()
    battle.user.name = "p1"
    battle.opponent.name = "p2"
    battle.user.active = Pokemon("weedle", level)
    battle.user.last_selected_move = LastUsedMove("weedle", "tackle", 0)
    battle.opponent.active = Pokemon(species, level)
    battle.team_inference.record_public_member(species, level)
    return battle


def _process(battle, *messages):
    battle.msg_list = list(messages)
    process_battle_updates(battle)


def _selected(context, evidence=None, *, species="pikachu", level=50, rng=None):
    return select_public_prior_variant(
        context,
        battle_format="gen9tugs",
        species_id=species,
        level=level,
        evidence=evidence,
        rng=rng,
    )


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class TestContextFallbackAndPrecedence(unittest.TestCase):
    def test_01_no_context_preserves_legacy_generic_sampling(self):
        battle = _battle()
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        generic.assert_called_once()

    def test_02_context_requires_explicit_fallback(self):
        dataset = _dataset()
        with self.assertRaises(TypeError):
            PublicPriorSearchContext(
                PublicPriorRegistry((dataset,)),
                (dataset.identity,),
                format_id="gen9tugs",
            )

    def test_03_generic_and_none_are_the_only_fallback_values(self):
        self.assertEqual(
            {"generic", "none"}, {item.value for item in PublicPriorFallback}
        )

    def test_04_public_context_is_immutable(self):
        context = _context()
        with self.assertRaises(FrozenInstanceError):
            context.format_id = "gen9ou"

    def test_05_registry_and_selected_identities_are_immutable(self):
        context = _context()
        self.assertIsInstance(context.selected_identities, tuple)
        with self.assertRaises(FrozenInstanceError):
            context.registry.datasets = ()

    def test_06_deepcopy_shares_immutable_public_records(self):
        context = _context()
        copied = copy.deepcopy(_battle(context))
        self.assertIs(context, copied.public_prior_context)
        self.assertIs(context.registry, copied.public_prior_context.registry)

    def test_07_two_battles_can_use_different_contexts(self):
        first = _context(_dataset("first"))
        second = _context(_dataset("second"))
        self.assertIs(first, _battle(first).public_prior_context)
        self.assertIs(second, _battle(second).public_prior_context)

    def test_08_global_standard_mode_has_no_public_registry(self):
        mode = StandardBattleMode()
        self.assertFalse(hasattr(mode, "public_prior_context"))
        self.assertFalse(
            any(isinstance(value, PublicPriorRegistry) for value in vars(mode).values())
        )

    def test_09_exact_format_match_permits_public_sampling(self):
        self.assertIs(PublicPriorSelectionStatus.SELECTED, _selected(_context()).status)

    def test_10_format_mismatch_prevents_public_sampling(self):
        result = select_public_prior_variant(
            _context(),
            battle_format="gen9ou",
            species_id="pikachu",
            level=50,
            evidence=None,
        )
        self.assertIs(PublicPriorSelectionStatus.CONTEXT_FORMAT_MISMATCH, result.status)

    def test_11_exact_forms_are_distinct(self):
        wash = _dataset(
            "wash", species_id="rotomwash", variants=(_variant(ability="levitate"),)
        )
        context = _context(wash)
        self.assertIs(
            PublicPriorSelectionStatus.SELECTED,
            _selected(context, species="rotomwash").status,
        )
        self.assertIs(
            PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT,
            _selected(context, species="rotomheat").status,
        )

    def test_12_dataset_selection_order_is_respected(self):
        first = _dataset("first", variants=(_variant(item="lightball"),))
        second = _dataset("second", variants=(_variant(item="choicescarf"),))
        context = _context(first, second, selected=(second.identity, first.identity))
        self.assertEqual("choicescarf", _selected(context).variant.item_id)

    def test_13_first_compatible_dataset_wins(self):
        miss = _dataset(
            "miss",
            variants=(_variant(moves=("tackle", "protect", "rest", "sleeptalk")),),
        )
        hit = _dataset("hit", variants=(_variant(item="choicespecs"),))
        context = _context(miss, hit)
        result = _selected(context, _evidence(moves=("thunderbolt",)))
        self.assertEqual(hit.identity, result.dataset_identity)

    def test_14_weights_are_not_merged_across_datasets(self):
        first = _dataset("first", variants=(_variant("only", weight=1),))
        second = _dataset("second", variants=(_variant("heavy", weight=10),))
        result = _selected(_context(first, second), rng=_FixedRng(0.99))
        self.assertEqual(first.identity, result.dataset_identity)

    def test_15_duplicate_selected_identities_are_rejected(self):
        dataset = _dataset()
        with self.assertRaises(ValueError):
            _context(dataset, selected=(dataset.identity, dataset.identity))

    def test_16_unknown_selected_identity_is_deterministic(self):
        dataset = _dataset()
        missing = PublicPriorIdentity("missing", "1", "gen9tugs")
        context = _context(dataset, selected=(missing, dataset.identity))
        self.assertIs(SelectedDatasetStatus.MISSING, context.diagnostics[0].status)
        self.assertEqual(dataset.identity, _selected(context).dataset_identity)

    def test_17_weighted_selection_uses_authored_weights_and_rng(self):
        variants = (_variant("a", weight=1), _variant("b", weight=3))
        self.assertEqual(
            "a", choose_weighted_public_variant(variants, _FixedRng(0)).variant_id
        )
        self.assertEqual(
            "b", choose_weighted_public_variant(variants, _FixedRng(0.99)).variant_id
        )

    def test_18_weighted_selection_does_not_mutate_weights(self):
        variants = (_variant("a", weight=2), _variant("b", weight=7))
        before = tuple(variant.weight for variant in variants)
        choose_weighted_public_variant(variants, _FixedRng(0.5))
        self.assertEqual(before, tuple(variant.weight for variant in variants))

    def test_18a_same_dataset_family_layers_one_weighted_pool(self):
        earlier_variant = _variant("earlier", weight=1, item="lightball")
        later_variant = _variant("later", weight=3, item="choicescarf")
        earlier = _dataset("family", version="2", variants=(earlier_variant,))
        later = _dataset("family", version="1", variants=(later_variant,))
        context = _context(
            earlier,
            later,
            selected=(earlier.identity, later.identity),
        )
        before = (earlier_variant.weight, later_variant.weight)

        first = _selected(context, rng=_FixedRng(0))
        last = _selected(context, rng=_FixedRng(0.99))

        self.assertEqual(
            (earlier.identity, "earlier"),
            (first.dataset_identity, first.variant.variant_id),
        )
        self.assertEqual(
            (later.identity, "later"), (last.dataset_identity, last.variant.variant_id)
        )
        self.assertEqual(before, (earlier_variant.weight, later_variant.weight))

    def test_18b_earlier_family_layer_shadows_duplicate_variant_id(self):
        earlier = _dataset(
            "family",
            version="2",
            variants=(_variant("duplicate", weight=1, item="lightball"),),
        )
        later = _dataset(
            "family",
            version="1",
            variants=(
                _variant(
                    "duplicate",
                    weight=100,
                    item="choicescarf",
                    moves=("quickattack", "irontail", "nuzzle", "protect"),
                ),
            ),
        )
        context = _context(
            earlier,
            later,
            selected=(earlier.identity, later.identity),
        )

        selected = _selected(context, rng=_FixedRng(0.99))
        shadowed_only = _selected(context, _evidence(moves=("quickattack",)))

        self.assertEqual(earlier.identity, selected.dataset_identity)
        self.assertEqual("lightball", selected.variant.item_id)
        self.assertIs(
            PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT,
            shadowed_only.status,
        )

    def test_18c_layered_family_applies_all_public_evidence_cumulatively(self):
        earlier = _dataset(
            "family",
            version="2",
            variants=(_variant("newer"),),
        )
        later = _dataset(
            "family",
            version="1",
            variants=(
                _variant(
                    "older",
                    item="choicescarf",
                    ability="lightningrod",
                    moves=("quickattack", "irontail", "nuzzle", "protect"),
                ),
            ),
        )
        context = _context(
            earlier,
            later,
            selected=(earlier.identity, later.identity),
        )
        evidence = _evidence(
            species="pikachu",
            level=50,
            moves=("quickattack", "nuzzle"),
            item="choicescarf",
            ability="lightningrod",
        )

        selected = _selected(context, evidence)

        self.assertEqual(later.identity, selected.dataset_identity)
        self.assertEqual("older", selected.variant.variant_id)
        self.assertIs(
            PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT,
            _selected(context, replace(evidence, species_id="raichu")).status,
        )
        self.assertIs(
            PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT,
            _selected(context, evidence, level=100).status,
        )


class TestCompatibilityAndFirewall(unittest.TestCase):
    def setUp(self):
        self.special = _variant("special")
        self.physical = _variant(
            "physical",
            item="choicescarf",
            ability="lightningrod",
            moves=("quickattack", "irontail", "nuzzle", "protect"),
        )
        self.context = _context(_dataset(variants=(self.special, self.physical)))

    def test_19_public_selected_move_filters_variants(self):
        self.assertEqual(
            "special",
            _selected(
                self.context, _evidence(moves=("thunderbolt",))
            ).variant.variant_id,
        )

    def test_20_multiple_moves_filter_cumulatively(self):
        evidence = _evidence(moves=("quickattack", "nuzzle"))
        self.assertEqual(
            "physical", _selected(self.context, evidence).variant.variant_id
        )

    def test_21_public_item_evidence_filters_variants(self):
        self.assertEqual(
            "physical",
            _selected(self.context, _evidence(item="choicescarf")).variant.variant_id,
        )

    def test_22_public_base_ability_filters_variants(self):
        evidence = _evidence(ability="lightningrod")
        self.assertEqual(
            "physical", _selected(self.context, evidence).variant.variant_id
        )

    def test_23_public_level_filters_variants(self):
        self.assertFalse(
            public_variant_is_compatible(self.special, "pikachu", 100, None)
        )

    def test_24_private_candidate_count_does_not_affect_sampling(self):
        battle = _battle(self.context)
        battle.team_inference._baseline_candidate_ids = ("private-a", "private-b")
        selected = _selected(battle.public_prior_context, rng=_FixedRng(0))
        self.assertEqual("physical", selected.variant.variant_id)

    def test_25_private_unique_match_does_not_affect_sampling(self):
        context = _context(_dataset(variants=(self.special,)))
        first = _battle(context)
        second = _battle(context)
        second.team_inference._baseline_candidate_ids = ("private-unique",)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            first_copy = prepare_battles(first, 1)[0][0]
            second_copy = prepare_battles(second, 1)[0][0]
        self.assertEqual(
            first_copy.opponent.active.item, second_copy.opponent.active.item
        )
        generic.assert_not_called()

    def test_26_private_exhausted_match_does_not_affect_sampling(self):
        context = _context(_dataset(variants=(self.special,)))
        battle = _battle(context)
        battle.team_inference._baseline_candidate_ids = ("private",)
        battle.team_inference._active_candidate_ids = ()
        battle.team_inference._filter_state = CandidateFilterState.EXHAUSTED
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        self.assertEqual("lightball", sampled.opponent.active.item)
        generic.assert_not_called()

    def test_27_search_source_does_not_inspect_private_candidate_ids(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "fp" / "search").glob("*.py")
        )
        forbidden = (
            "TeamPoolCandidateId",
            "TeamRecordId",
            "PoolIdentity",
            "baseline_candidate_ids",
            "active_candidate_ids",
            "CandidateAccess",
        )
        self.assertFalse({token for token in forbidden if token in sources})

    def test_28_public_evidence_records_without_private_pool(self):
        battle = _battle()
        member = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual(("pikachu", 50), (member.species_id, member.level))
        self.assertEqual(0, battle.team_inference.baseline_candidate_count)

    def test_29_selected_moves_record_without_private_match(self):
        battle = _battle()
        _process(battle, "|move|p2a: Pikachu|Thunderbolt|p1a: Weedle")
        member = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual(("thunderbolt",), member.selected_move_ids)

    def test_30_closing_jaws_moves_record_without_private_match(self):
        battle = _battle()
        _process(
            battle,
            "|move|p2a: Pikachu|Iron Head|p1a: Weedle|[from] ability: Closing Jaws",
        )
        member = battle.team_inference.observation_ledger.member("pikachu")
        self.assertIn(
            PublicObservationSource.CLOSING_JAWS_SELECTED_MOVE, member.provenance
        )

    def test_31_corrosive_gas_item_evidence_records_without_private_match(self):
        battle = _battle()
        battle.opponent.active.item = "lightball"
        _process(battle, "|-enditem|p2a: Pikachu|Light Ball|[from] move: Corrosive Gas")
        member = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("lightball", member.initial_item_id)

    def test_32_trace_evidence_records_without_private_match(self):
        battle = _battle()
        _process(
            battle,
            "|-ability|p2a: Pikachu|Closing Jaws|Trace|[from] ability: Trace|[of] p1a: Weedle",
        )
        member = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("trace", member.base_ability_id)
        self.assertTrue(member.current_ability_changed)

    def test_32a_hidden_power_move_compatibility_matrix(self):
        compatible = (
            ("hiddenpower", "hiddenpowerfire60"),
            ("hiddenpower", "hiddenpowergrass60"),
            ("hiddenpower", "hiddenpowerice60"),
            ("hiddenpowerfire", "hiddenpowerfire60"),
            ("hiddenpowerfire60", "hiddenpowerfire"),
            ("Hidden Power Fire", "HIDDEN-POWER-FIRE-60"),
            ("hiddenpowerwater70", "hiddenpowerwater60"),
            ("Thunder Bolt", "thunderbolt"),
        )
        incompatible = (
            ("hiddenpower", "flamethrower"),
            ("flamethrower", "hiddenpowerfire60"),
            ("hiddenpowerfire", "hiddenpowergrass60"),
            ("hiddenpowerfire60", "hiddenpowerice60"),
            ("hiddenpowergrass", "hiddenpowerice60"),
            ("hiddenpowerfireblast", "hiddenpowerfire60"),
            ("hiddenpowerfairy", "hiddenpowerfire60"),
            ("hiddenpower60", "hiddenpowerfire60"),
            ("thunderbolt", "voltswitch"),
        )
        for observed, candidate in compatible:
            with self.subTest(observed=observed, candidate=candidate):
                self.assertTrue(
                    observed_move_is_compatible_with_candidate_move(observed, candidate)
                )
        for observed, candidate in incompatible:
            with self.subTest(observed=observed, candidate=candidate):
                self.assertFalse(
                    observed_move_is_compatible_with_candidate_move(observed, candidate)
                )

    def test_32b_generic_hidden_power_keeps_all_typed_candidates_only(self):
        variants = (
            _variant("fire", moves=("hiddenpowerfire60", "surf", "protect", "rest")),
            _variant("grass", moves=("hiddenpowergrass60", "surf", "protect", "rest")),
            _variant("ice", moves=("hiddenpowerice60", "surf", "protect", "rest")),
            _variant("ordinary", moves=("flamethrower", "surf", "protect", "rest")),
        )
        evidence = _evidence(moves=("hiddenpower",))
        compatible_ids = {
            variant.variant_id
            for variant in variants
            if public_variant_is_compatible(variant, "pikachu", 50, evidence)
        }
        self.assertEqual({"fire", "grass", "ice"}, compatible_ids)

    def test_32c_typed_hidden_power_narrows_non_destructively(self):
        variants = (
            _variant("fire", moves=("hiddenpowerfire60", "surf", "protect", "rest")),
            _variant("grass", moves=("hiddenpowergrass60", "surf", "protect", "rest")),
            _variant("ice", moves=("hiddenpowerice60", "surf", "protect", "rest")),
        )
        context = _context(_dataset(variants=variants))
        initial = _evidence(
            moves=("hiddenpower", "surf"), item="lightball", ability="static"
        )
        self.assertTrue(
            all(
                public_variant_is_compatible(variant, "pikachu", 50, initial)
                for variant in variants
            )
        )
        narrowed = replace(
            initial,
            selected_move_ids=("hiddenpower", "hiddenpowerfire", "surf"),
        )
        result = _selected(context, narrowed)
        self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
        self.assertEqual("fire", result.variant.variant_id)
        self.assertEqual(initial.initial_item_id, narrowed.initial_item_id)
        self.assertEqual(initial.base_ability_id, narrowed.base_ability_id)
        self.assertIn("hiddenpower", narrowed.selected_move_ids)
        self.assertIn("surf", narrowed.selected_move_ids)

    def test_32d_protocol_preserves_generic_and_typed_ledger_evidence(self):
        generic = _battle()
        _process(
            generic,
            "|move|p2a: Pikachu|Hidden Power|p1a: Weedle",
            "|-damage|p1a: Weedle|90/100",
        )
        self.assertEqual(
            ("hiddenpower",),
            generic.team_inference.observation_ledger.member(
                "pikachu"
            ).selected_move_ids,
        )

        typed = _battle()
        _process(typed, "|move|p2a: Pikachu|Hidden Power Fire|p1a: Weedle")
        self.assertEqual(
            ("hiddenpowerfire",),
            typed.team_inference.observation_ledger.member("pikachu").selected_move_ids,
        )


class TestOriginalItemTransitions(unittest.TestCase):
    def test_confident_original_item_compatibility_matrix(self):
        cases = (
            ("throatspray", "throatspray", True),
            ("choicespecs", "throatspray", False),
            ("throatspray", "choicespecs", False),
            ("focussash", "focussash", True),
            ("leftovers", "focussash", False),
            ("eviolite", "eviolite", True),
            ("leftovers", "eviolite", False),
            ("heavydutyboots", "eviolite", False),
            ("leftovers", "leftovers", True),
            ("rockyhelmet", "leftovers", False),
            ("heavydutyboots", "leftovers", False),
            ("lightball", "lightball", True),
        )
        for candidate, observed, expected in cases:
            with self.subTest(candidate=candidate, observed=observed):
                self.assertIs(
                    expected,
                    candidate_original_item_is_compatible(
                        candidate, _evidence(item=observed)
                    ),
                )

    def test_item_id_normalization_stays_in_protocol_observation_path(self):
        battle = _battle()
        _process(battle, "|-item|p2a: Pikachu|Throat Spray")
        evidence = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("throatspray", evidence.initial_item_id)
        self.assertTrue(candidate_original_item_is_compatible("throatspray", evidence))

    def test_confident_original_item_precedes_an_inconsistent_ambiguity_flag(self):
        evidence = _evidence(item="throatspray", ambiguous=True)
        self.assertTrue(candidate_original_item_is_compatible("throatspray", evidence))
        self.assertFalse(candidate_original_item_is_compatible("choicespecs", evidence))

    def test_bare_enditem_preserves_throat_spray_original_item(self):
        battle = _battle()
        battle.opponent.active.item = constants.UNKNOWN_ITEM
        _process(
            battle,
            "|move|p2a: Pikachu|Boomburst|p1a: Weedle",
            "|-enditem|p2a: Pikachu|Throat Spray",
        )
        evidence = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("throatspray", evidence.initial_item_id)
        self.assertIn(PublicObservationSource.ITEM_REMOVED, evidence.provenance)
        self.assertIsNone(battle.opponent.active.item)
        self.assertEqual("throatspray", battle.opponent.active.removed_item)
        self.assertTrue(candidate_original_item_is_compatible("throatspray", evidence))
        self.assertFalse(candidate_original_item_is_compatible("choicespecs", evidence))

    def test_focus_sash_consumption_preserves_original_item(self):
        battle = _battle()
        battle.opponent.active.item = "focussash"
        _process(battle, "|-enditem|p2a: Pikachu|Focus Sash|[consumed]")
        evidence = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("focussash", evidence.initial_item_id)
        self.assertIn(PublicObservationSource.ITEM_CONSUMED, evidence.provenance)
        self.assertIsNone(battle.opponent.active.item)
        self.assertFalse(candidate_original_item_is_compatible("leftovers", evidence))

    def test_knock_off_and_corrosive_gas_preserve_original_item(self):
        for move_name in ("Knock Off", "Corrosive Gas"):
            with self.subTest(move=move_name):
                battle = _battle()
                battle.opponent.active.item = "eviolite"
                _process(
                    battle,
                    "|-enditem|p2a: Pikachu|Eviolite|[from] move: " + move_name,
                )
                evidence = battle.team_inference.observation_ledger.member("pikachu")
                self.assertEqual("eviolite", evidence.initial_item_id)
                self.assertIsNone(battle.opponent.active.item)
                self.assertTrue(
                    candidate_original_item_is_compatible("eviolite", evidence)
                )
                self.assertFalse(
                    candidate_original_item_is_compatible("leftovers", evidence)
                )

    def test_transfer_preserves_confident_original_and_acquired_current_item(self):
        battle = _battle()
        battle.opponent.active.item = "leftovers"
        _process(battle, "|-item|p2a: Pikachu|Leftovers")
        _process(
            battle,
            "|-item|p2a: Pikachu|Choice Scarf|[from] move: Trick",
        )
        evidence = battle.team_inference.observation_ledger.member("pikachu")
        self.assertEqual("leftovers", evidence.initial_item_id)
        self.assertFalse(evidence.item_ambiguous)
        self.assertIn(PublicObservationSource.ITEM_ACQUIRED, evidence.provenance)
        self.assertEqual("choicescarf", battle.opponent.active.item)
        self.assertEqual("leftovers", battle.opponent.active.removed_item)
        self.assertTrue(candidate_original_item_is_compatible("leftovers", evidence))
        self.assertFalse(candidate_original_item_is_compatible("choicescarf", evidence))

    def test_acquisition_without_original_evidence_remains_ambiguous(self):
        battle = _battle()
        _process(
            battle,
            "|-item|p2a: Pikachu|Choice Scarf|[from] move: Switcheroo",
        )
        evidence = battle.team_inference.observation_ledger.member("pikachu")
        self.assertIsNone(evidence.initial_item_id)
        self.assertTrue(evidence.item_ambiguous)
        self.assertTrue(candidate_original_item_is_compatible("leftovers", evidence))

    def test_item_suppression_changes_neither_original_nor_current_item(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = "leftovers"
        evidence = _evidence(item="leftovers")
        populate_pokemon_from_public_variant(
            pokemon, _variant(item="leftovers"), evidence
        )
        self.assertEqual("leftovers", pokemon.item)
        self.assertTrue(candidate_original_item_is_compatible("leftovers", evidence))

    def test_unknown_current_item_does_not_erase_confident_original_item(self):
        pokemon = Pokemon("pikachu", 50)
        evidence = _evidence(item="lightball")
        populate_pokemon_from_public_variant(pokemon, _variant(), evidence)
        self.assertEqual("lightball", evidence.initial_item_id)
        self.assertEqual("lightball", pokemon.item)

    def test_known_no_item_does_not_admit_other_original_items(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = None
        pokemon.removed_item = "throatspray"
        evidence = _evidence(item="throatspray")
        self.assertFalse(candidate_original_item_is_compatible("choicespecs", evidence))
        populate_pokemon_from_public_variant(
            pokemon, _variant(item="throatspray"), evidence
        )
        self.assertIsNone(pokemon.item)

    def test_repeated_item_transition_evidence_is_idempotent(self):
        battle = _battle()
        battle.opponent.active.item = "leftovers"
        message = "|-enditem|p2a: Pikachu|Leftovers|[from] move: Knock Off"
        _process(battle, message)
        first = battle.team_inference.observation_ledger
        _process(battle, message)
        self.assertIs(first, battle.team_inference.observation_ledger)

    def test_consumed_and_removed_items_are_never_restored(self):
        for item in ("throatspray", "focussash", "eviolite", "leftovers"):
            with self.subTest(item=item):
                pokemon = Pokemon("pikachu", 50)
                pokemon.item = None
                pokemon.removed_item = item
                variant = _variant(item=item)
                populate_pokemon_from_public_variant(
                    pokemon, variant, _evidence(item=item)
                )
                self.assertIsNone(pokemon.item)
                self.assertEqual(item, pokemon.removed_item)

    def test_acquired_replacement_item_is_not_overwritten(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = "choicescarf"
        pokemon.removed_item = "leftovers"
        evidence = _evidence(item="leftovers")
        populate_pokemon_from_public_variant(
            pokemon, _variant(item="leftovers"), evidence
        )
        self.assertEqual("choicescarf", pokemon.item)
        self.assertEqual("leftovers", evidence.initial_item_id)

    def test_candidate_population_remains_coherent_and_serializable(self):
        variant = _variant(item="focussash")
        context = _context(_dataset(variants=(variant,)))
        battle = _battle(context)
        battle.opponent.active.item = None
        battle.opponent.active.removed_item = "focussash"
        battle.team_inference.record_initial_item(
            "pikachu", "focussash", PublicObservationSource.ITEM_CONSUMED
        )
        before = copy.deepcopy(battle.opponent.active)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        copied = sampled.opponent.active
        self.assertIsNone(copied.item)
        self.assertEqual(set(variant.move_ids), {move.name for move in copied.moves})
        self.assertEqual(variant.base_ability_id, copied.ability)
        self.assertIsInstance(battle_to_poke_engine_state(sampled).to_string(), str)
        self.assertEqual(before, battle.opponent.active)

    def test_public_success_and_fallback_policy_follow_real_contradictions(self):
        variant = _variant(item="lightball")
        for fallback, expected_generic_calls in (
            (PublicPriorFallback.NONE, 0),
            (PublicPriorFallback.GENERIC, 1),
        ):
            with self.subTest(fallback=fallback.value):
                context = _context(_dataset(variants=(variant,)), fallback=fallback)
                battle = _battle(context)
                battle.team_inference.record_initial_item(
                    "pikachu",
                    "sitrusberry",
                    PublicObservationSource.DIRECT_ITEM_REVEAL,
                )
                battle.opponent.active.add_move("thunderbolt")
                battle.opponent.active.ability = "static"
                canonical = copy.deepcopy(battle.opponent.active)
                with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
                    sampled = prepare_battles(battle, 1)[0][0]
                self.assertEqual(expected_generic_calls, generic.call_count)
                self.assertEqual(canonical, battle.opponent.active)
                self.assertEqual("thunderbolt", sampled.opponent.active.moves[0].name)
                self.assertEqual("static", sampled.opponent.active.ability)


class TestCopiedPopulationAndFallback(unittest.TestCase):
    def setUp(self):
        self.variant = _variant(ivs=_stats(atk=0, spe=0))
        self.context = _context(_dataset(variants=(self.variant,)))

    def _sample(self, battle=None, count=1):
        battle = battle or _battle(self.context)
        return prepare_battles(battle, count)

    def test_33_variant_populates_only_copied_battle(self):
        battle = _battle(self.context)
        sampled = self._sample(battle)[0][0]
        self.assertEqual(constants.UNKNOWN_ITEM, battle.opponent.active.item)
        self.assertEqual("lightball", sampled.opponent.active.item)

    def test_34_canonical_opponent_item_remains_unchanged(self):
        battle = _battle(self.context)
        before = battle.opponent.active.item
        self._sample(battle)
        self.assertEqual(before, battle.opponent.active.item)

    def test_35_canonical_opponent_ability_remains_unchanged(self):
        battle = _battle(self.context)
        self._sample(battle)
        self.assertIsNone(battle.opponent.active.ability)

    def test_36_canonical_opponent_moves_remain_unchanged(self):
        battle = _battle(self.context)
        battle.opponent.active.add_move("thunderbolt")
        before = tuple(battle.opponent.active.moves)
        self._sample(battle)
        self.assertEqual(before, tuple(battle.opponent.active.moves))

    def test_37_canonical_spread_remains_unchanged(self):
        battle = _battle(self.context)
        before = (
            battle.opponent.active.nature,
            battle.opponent.active.evs,
            battle.opponent.active.ivs,
        )
        self._sample(battle)
        after = (
            battle.opponent.active.nature,
            battle.opponent.active.evs,
            battle.opponent.active.ivs,
        )
        self.assertEqual(before, after)

    def test_38_sampled_copy_receives_one_coherent_variant(self):
        pokemon = self._sample()[0][0].opponent.active
        self.assertEqual(
            ("lightball", "static", "timid"),
            (pokemon.item, pokemon.ability, pokemon.nature),
        )
        self.assertEqual(
            set(self.variant.move_ids), {move.name for move in pokemon.moves}
        )
        self.assertEqual(self.variant.ivs.as_tuple(), pokemon.ivs)

    def test_39_generic_sampler_not_called_after_public_success(self):
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            self._sample()
        generic.assert_not_called()

    def test_40_generic_calls_existing_sampler_after_public_miss(self):
        context = _context(_dataset(), fallback=PublicPriorFallback.GENERIC)
        battle = _battle(context, species="raichu")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        generic.assert_called_once()

    def test_41_none_does_not_call_generic_after_public_miss(self):
        battle = _battle(_context(), species="raichu")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        generic.assert_not_called()

    def test_42_public_exhaustion_does_not_crash_search(self):
        battle = _battle(self.context)
        battle.team_inference.record_selected_move("pikachu", "tackle")
        sampled = self._sample(battle)[0][0]
        self.assertEqual([], sampled.opponent.active.moves)

    def test_43_fainted_pokemon_are_not_sampled(self):
        battle = _battle(self.context)
        battle.opponent.active.hp = 0
        with mock.patch(
            "fp.search.standard_battles.populate_pokemon_from_public_variant"
        ) as populate:
            self._sample(battle)
        populate.assert_not_called()

    def test_44_own_pokemon_are_not_sampled_from_opponent_priors(self):
        battle = _battle(self.context)
        before = copy.deepcopy(battle.user.active)
        sampled = self._sample(battle)[0][0]
        self.assertEqual(
            (before.item, before.ability, before.moves),
            (
                sampled.user.active.item,
                sampled.user.active.ability,
                sampled.user.active.moves,
            ),
        )

    def test_45_revealed_move_pp_is_preserved(self):
        battle = _battle(self.context)
        move = battle.opponent.active.add_move("thunderbolt")
        move.current_pp = 3
        sampled_move = self._sample(battle)[0][0].opponent.active.get_move(
            "thunderbolt"
        )
        self.assertEqual(
            (3, move.max_pp), (sampled_move.current_pp, sampled_move.max_pp)
        )

    def test_46_missing_variant_moves_are_added_once(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.add_move("thunderbolt")
        self.assertTrue(
            populate_pokemon_from_public_variant(
                pokemon, self.variant, _evidence(moves=("thunderbolt",))
            )
        )
        self.assertEqual(4, len({move.name for move in pokemon.moves}))

    def test_47_publicly_revealed_moves_are_never_removed(self):
        pokemon = Pokemon("pikachu", 50)
        revealed = pokemon.add_move("thunderbolt")
        populate_pokemon_from_public_variant(
            pokemon, self.variant, _evidence(moves=("thunderbolt",))
        )
        self.assertIs(revealed, pokemon.get_move("thunderbolt"))

    def test_48_five_move_states_are_never_constructed(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.add_move("tackle")
        before = tuple(pokemon.moves)
        self.assertFalse(
            populate_pokemon_from_public_variant(pokemon, self.variant, None)
        )
        self.assertEqual(before, tuple(pokemon.moves))

    def test_49_removed_item_is_not_restored(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = None
        pokemon.removed_item = "lightball"
        populate_pokemon_from_public_variant(
            pokemon, self.variant, _evidence(item="lightball")
        )
        self.assertEqual((None, "lightball"), (pokemon.item, pokemon.removed_item))

    def test_50_consumed_item_is_not_restored(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = None
        pokemon.removed_item = "sitrusberry"
        populate_pokemon_from_public_variant(
            pokemon, self.variant, _evidence(item="sitrusberry")
        )
        self.assertIsNone(pokemon.item)

    def test_51_transferred_current_item_is_preserved(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.item = "choicescarf"
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertEqual("choicescarf", pokemon.item)

    def test_52_unknown_current_item_may_be_populated(self):
        pokemon = Pokemon("pikachu", 50)
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertEqual("lightball", pokemon.item)

    def test_53_known_base_and_current_ability_are_preserved(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.ability = pokemon.original_ability = "static"
        populate_pokemon_from_public_variant(
            pokemon, self.variant, _evidence(ability="static")
        )
        self.assertEqual(
            ("static", "static"), (pokemon.original_ability, pokemon.ability)
        )

    def test_54_trace_base_and_copied_current_ability_are_preserved(self):
        trace_variant = _variant(ability="trace")
        pokemon = Pokemon("pikachu", 50)
        pokemon.original_ability = "trace"
        pokemon.ability = "closingjaws"
        evidence = _evidence(ability="trace", changed=True)
        populate_pokemon_from_public_variant(pokemon, trace_variant, evidence)
        self.assertEqual(
            ("trace", "closingjaws"), (pokemon.original_ability, pokemon.ability)
        )

    def test_55_gastro_acid_suppression_remains_present(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.volatile_statuses.append("gastroacid")
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertIn("gastroacid", pokemon.volatile_statuses)

    def test_56_neutralizing_gas_runtime_state_remains_present(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.volatile_statuses.append("neutralizinggas")
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertIn("neutralizinggas", pokemon.volatile_statuses)

    def test_57_hp_proportion_is_preserved_after_recalculation(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.hp = pokemon.max_hp // 2
        before = pokemon.hp / pokemon.max_hp
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertAlmostEqual(before, pokemon.hp / pokemon.max_hp, delta=0.01)

    def test_58_status_is_preserved(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.status = constants.Status.BURN
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertIs(constants.Status.BURN, pokemon.status)

    def test_59_boosts_are_preserved(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.boosts[constants.ATTACK] = 2
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertEqual(2, pokemon.boosts[constants.ATTACK])

    def test_60_volatiles_are_preserved(self):
        pokemon = Pokemon("pikachu", 50)
        pokemon.volatile_statuses.append("substitute")
        populate_pokemon_from_public_variant(pokemon, self.variant, None)
        self.assertIn("substitute", pokemon.volatile_statuses)

    def test_60a_generic_hidden_power_resolves_only_in_copied_state(self):
        variant = _variant(
            "typed",
            moves=("hiddenpowerfire60", "surf", "protect", "rest"),
            ivs=_stats(atk=0, spe=0),
        )
        context = _context(
            _dataset(variants=(variant,)), fallback=PublicPriorFallback.GENERIC
        )
        battle = _battle(context)
        _process(
            battle,
            "|move|p2a: Pikachu|Hidden Power|p1a: Weedle",
            "|-damage|p1a: Weedle|90/100",
        )
        canonical_move = battle.opponent.active.get_move("hiddenpower")
        canonical_move.current_pp = 7
        canonical_before = copy.deepcopy(battle.opponent.active)
        ledger_before = battle.team_inference.observation_ledger

        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]

        generic.assert_not_called()
        sampled_pokemon = sampled.opponent.active
        self.assertEqual(
            "typed",
            _selected(context, ledger_before.member("pikachu")).variant.variant_id,
        )
        self.assertEqual(
            set(variant.move_ids), {move.name for move in sampled_pokemon.moves}
        )
        self.assertEqual(7, sampled_pokemon.get_move("hiddenpowerfire60").current_pp)
        self.assertEqual(
            (variant.item_id, variant.base_ability_id, variant.nature_id),
            (
                sampled_pokemon.item,
                sampled_pokemon.ability,
                sampled_pokemon.nature,
            ),
        )
        self.assertEqual(variant.evs.as_tuple(), tuple(sampled_pokemon.evs))
        self.assertEqual(variant.ivs.as_tuple(), sampled_pokemon.ivs)
        self.assertEqual(variant.level, sampled_pokemon.level)
        self.assertIsInstance(battle_to_poke_engine_state(sampled).to_string(), str)
        self.assertIs(ledger_before, battle.team_inference.observation_ledger)
        self.assertEqual(
            ("hiddenpower",),
            battle.team_inference.observation_ledger.member(
                "pikachu"
            ).selected_move_ids,
        )
        self.assertEqual("hiddenpower", battle.opponent.active.moves[0].name)
        self.assertEqual(7, battle.opponent.active.moves[0].current_pp)
        self.assertEqual(
            (
                canonical_before.item,
                canonical_before.ability,
                canonical_before.nature,
                canonical_before.evs,
                canonical_before.ivs,
                canonical_before.moves,
            ),
            (
                battle.opponent.active.item,
                battle.opponent.active.ability,
                battle.opponent.active.nature,
                battle.opponent.active.evs,
                battle.opponent.active.ivs,
                battle.opponent.active.moves,
            ),
        )


class TestIVSerializationAndIsolation(unittest.TestCase):
    def setUp(self):
        self.full = _variant("full", ivs=_stats())
        self.zero = _variant("zero", ivs=_stats(atk=0, spe=0))

    def test_61_default_ivs_remain_31_for_legacy_pokemon(self):
        self.assertEqual((31,) * 6, Pokemon("pikachu", 50).ivs)

    def test_62_zero_attack_iv_changes_sampled_attack(self):
        full = Pokemon("pikachu", 50)
        zero = Pokemon("pikachu", 50)
        populate_pokemon_from_public_variant(full, self.full, None)
        populate_pokemon_from_public_variant(zero, self.zero, None)
        self.assertLess(zero.stats[constants.ATTACK], full.stats[constants.ATTACK])

    def test_63_zero_speed_iv_changes_sampled_speed(self):
        full = Pokemon("pikachu", 50)
        zero = Pokemon("pikachu", 50)
        populate_pokemon_from_public_variant(full, self.full, None)
        populate_pokemon_from_public_variant(zero, self.zero, None)
        self.assertLess(zero.stats[constants.SPEED], full.stats[constants.SPEED])

    def test_64_public_ivs_affect_serialized_stats(self):
        full = Pokemon("pikachu", 50)
        zero = Pokemon("pikachu", 50)
        populate_pokemon_from_public_variant(full, self.full, None)
        populate_pokemon_from_public_variant(zero, self.zero, None)
        self.assertLess(
            pokemon_to_poke_engine_pkmn(zero).speed,
            pokemon_to_poke_engine_pkmn(full).speed,
        )

    def test_65_no_poke_engine_state_schema_field_is_added(self):
        state = battle_to_poke_engine_state(_battle())
        self.assertNotIn("public_prior", state.to_string())
        self.assertFalse(hasattr(state, "public_prior_context"))

    def test_66_generic_sampling_retains_default_iv_behavior(self):
        battle = _battle()

        def legacy_generic(pokemon, mode):
            pokemon.set_spread("timid", [0, 0, 0, 252, 4, 252])

        with mock.patch(
            "fp.search.standard_battles.sample_pokemon", side_effect=legacy_generic
        ):
            sampled = prepare_battles(battle, 1)[0][0]
        self.assertEqual((31,) * 6, sampled.opponent.active.ivs)
        self.assertIsInstance(sampled.opponent.active.evs, list)

    def test_67_multiple_prepared_samples_do_not_mutate_one_another(self):
        sampled = prepare_battles(_battle(_context(_dataset(variants=(self.zero,)))), 2)
        first, second = sampled[0][0], sampled[1][0]
        first.opponent.active.item = "changed"
        self.assertEqual("lightball", second.opponent.active.item)

    def test_68_public_registry_remains_immutable_after_sampling(self):
        context = _context(_dataset(variants=(self.zero,)))
        before = context.registry.datasets
        prepare_battles(_battle(context), 1)
        self.assertEqual(before, context.registry.datasets)

    def test_69_public_ledger_remains_immutable_after_sampling(self):
        battle = _battle(_context(_dataset(variants=(self.zero,))))
        before = battle.team_inference.observation_ledger
        prepare_battles(battle, 1)
        self.assertIs(before, battle.team_inference.observation_ledger)

    def test_70_search_sample_weights_keep_existing_behavior(self):
        samples = prepare_battles(_battle(_context(_dataset(variants=(self.zero,)))), 4)
        self.assertEqual([0.25] * 4, [weight for _, weight in samples])

    def test_71_mcts_input_remains_ordinary_serialized_state(self):
        sampled = prepare_battles(
            _battle(_context(_dataset(variants=(self.zero,)))), 1
        )[0][0]
        state_string = battle_to_poke_engine_state(sampled).to_string()
        self.assertIsInstance(state_string, str)
        self.assertNotIn("syntheticprior", state_string)

    def test_72_process_workers_do_not_need_public_registry_access(self):
        source = (ROOT / "fp" / "search" / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("PublicPriorRegistry", source)
        self.assertNotIn("public_prior_context", source)

    def test_73_no_runtime_cache_is_created(self):
        context = _context()
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("runtime file access")
        ):
            self.assertIs(
                PublicPriorSelectionStatus.SELECTED, _selected(context).status
            )

    def test_74_no_network_call_occurs(self):
        context = _context()
        with mock.patch.object(
            socket, "create_connection", side_effect=AssertionError("network")
        ):
            self.assertIs(
                PublicPriorSelectionStatus.SELECTED, _selected(context).status
            )

    def test_75_no_production_public_prior_json_exists(self):
        self.assertEqual(
            [], list((ROOT / "fp" / "data" / "public_priors").glob("*.json"))
        )

    def test_76_no_private_team_pool_conversion_path_exists(self):
        paths = (
            ROOT / "fp" / "search" / "public_prior_sampling.py",
            ROOT / "fp" / "battle" / "public_prior_context.py",
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("fp.data.team_pools", source)
        self.assertNotIn("PokemonRecord", source)

    def test_77_ordinary_gen9_sampling_is_unchanged(self):
        battle = _battle(format_id="gen9ou")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        generic.assert_called_once()

    def test_78_ordinary_national_dex_sampling_is_unchanged(self):
        battle = _battle(format_id="gen9nationaldex")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        generic.assert_called_once()


class TestCumulativeRegressions(unittest.TestCase):
    def test_79_phase_one_loader_checkpoint_remains_present(self):
        source = (ROOT / "tests" / "test_team_pool_loader.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class Test", source)
        self.assertTrue(
            (ROOT / "fp" / "data" / "team_pools" / "validation.py").is_file()
        )

    def test_80_phase_two_inference_checkpoint_remains_present(self):
        battle = _battle()
        self.assertEqual(0, battle.team_inference.baseline_candidate_count)
        self.assertFalse(battle.team_inference.exact_fields_available)

    def test_81_phase_three_public_ledger_checkpoint_remains_present(self):
        ledger = _battle().team_inference.observation_ledger
        with self.assertRaises(FrozenInstanceError):
            ledger.revision = 2

    def test_82_phase_four_public_prior_checkpoint_remains_present(self):
        dataset = _dataset()
        self.assertEqual("public", dataset.visibility)
        with self.assertRaises(FrozenInstanceError):
            dataset.visibility = "private"

    def test_83_crag_mend_behavior_remains_passing(self):
        moves_before = copy.deepcopy(all_move_json)
        pokedex_before = copy.deepcopy(pokedex)
        try:
            apply_mods(FormatSpec.from_format_string("gen9tugs"))
            self.assertEqual(5, all_move_json["cragmend"][constants.PP])
            self.assertTrue(unlikely_to_have_choice_item("cragmend"))
        finally:
            all_move_json.clear()
            all_move_json.update(moves_before)
            pokedex.clear()
            pokedex.update(pokedex_before)

    def test_84_trick_room_persistent_behavior_remains_passing(self):
        battle = _battle()
        battle.user.active = Pokemon("lapras", 50)
        battle.user.active.ability = "persistent"
        fieldstart(
            battle,
            ["", "-fieldstart", "move: Trick Room", "[of] p1a: Lapras", "[persistent]"],
        )
        self.assertEqual(8, battle.trick_room_turns_remaining)

    def test_85_ancient_shell_serialization_remains_passing(self):
        pokemon = Pokemon("lapras", 50)
        pokemon.ability = "ancientshell"
        self.assertEqual("ancientshell", pokemon_to_poke_engine_pkmn(pokemon).ability)

    def test_86_corrosive_gas_parsing_remains_passing(self):
        battle = _battle(species="dustox")
        battle.opponent.active.item = "leftovers"
        remove_item(
            battle,
            ["", "-enditem", "p2a: Dustox", "Leftovers", "[from] move: Corrosive Gas"],
        )
        self.assertIsNone(battle.opponent.active.item)
        self.assertEqual("leftovers", battle.opponent.active.removed_item)

    def test_87_closing_jaws_bookkeeping_remains_passing(self):
        battle = _battle(species="mawile")
        activate(battle, ["", "-activate", "p2a: Mawile", "ability: Closing Jaws"])
        self.assertEqual("closingjaws", battle.opponent.active.ability)

    def test_88_transformed_runtime_state_is_left_intact(self):
        variant = _variant()
        pokemon = Pokemon("pikachu", 50)
        pokemon.volatile_statuses.append(constants.TRANSFORM)
        pokemon.item = "choicescarf"
        pokemon.ability = "levitate"
        pokemon.moves = [copy.deepcopy(Pokemon("weedle", 50).add_move("tackle"))]
        before = copy.deepcopy(pokemon)
        self.assertTrue(populate_pokemon_from_public_variant(pokemon, variant, None))
        self.assertEqual(
            (before.item, before.ability, before.nature, before.ivs, before.moves),
            (pokemon.item, pokemon.ability, pokemon.nature, pokemon.ivs, pokemon.moves),
        )

    def test_89_active_and_reserve_placement_is_preserved(self):
        context = _context()
        battle = _battle(context)
        battle.opponent.reserve.append(Pokemon("raichu", 50))
        battle.team_inference.record_public_member("raichu", 50)
        sampled = prepare_battles(battle, 1)[0][0]
        self.assertEqual("pikachu", sampled.opponent.active.name)
        self.assertEqual(
            ["raichu"], [pokemon.name for pokemon in sampled.opponent.reserve]
        )

    def test_90_ambiguous_item_history_blocks_item_injection(self):
        pokemon = Pokemon("pikachu", 50)
        evidence = _evidence(ambiguous=True)
        populate_pokemon_from_public_variant(pokemon, _variant(), evidence)
        self.assertEqual(constants.UNKNOWN_ITEM, pokemon.item)

    def test_91_conflicting_public_evidence_exhausts_variants(self):
        result = _selected(_context(), _evidence(conflict=True))
        self.assertIs(PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT, result.status)

    def test_92_own_authored_zero_ivs_are_metadata_only(self):
        battle = Battle("own-ivs")
        battle.user.team_dict = [
            {
                "species": "pikachu",
                "nature": "timid",
                "evs": {stat: 0 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
                "ivs": {"hp": 31, "atk": 0, "def": 31, "spa": 31, "spd": 31, "spe": 0},
            }
        ]
        request = {
            "active": [{"moves": [{"id": "tackle", "move": "Tackle", "pp": 35}]}],
            "side": {
                "id": "p1",
                "pokemon": [
                    {
                        "ident": "p1: Pikachu",
                        "details": "Pikachu, L50",
                        "condition": "100/100",
                        "active": True,
                        "stats": {
                            "atk": 77,
                            "def": 60,
                            "spa": 70,
                            "spd": 65,
                            "spe": 90,
                        },
                        "moves": ["tackle"],
                        "baseAbility": "static",
                        "ability": "static",
                        "item": "lightball",
                    }
                ],
            },
        }
        battle.user.initialize_first_turn_user_from_json(request)
        self.assertEqual((31, 0, 31, 31, 31, 0), battle.user.active.ivs)
        self.assertEqual(77, battle.user.active.stats[constants.ATTACK])


if __name__ == "__main__":
    unittest.main()
