import builtins
import copy
import socket
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from fp import constants
from fp.battle.protocol import (
    activate,
    fieldstart,
    process_battle_updates,
    remove_item,
    unlikely_to_have_choice_item,
)
from fp.battle.helpers import normalize_name
from fp.battle.state import Battle, LastUsedMove
from fp.battle.team_inference import (
    CandidateAccess,
    CandidateFilterState,
    OpponentObservationLedger,
    PublicObservationSource,
    TeamPoolMatchState,
    TeamSheetPolicy,
    filter_team_candidates,
)
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.sets import SmogonSets, TeamDatasets
from fp.data.team_pools import (
    PokemonRecord,
    StatValues,
    TeamPool,
    TeamPoolRegistry,
    TeamRecord,
)
from fp.format_spec import FormatSpec
from fp.modes.standard_battle import StandardBattleMode
from fp.search.poke_engine_helpers import (
    battle_to_poke_engine_state,
    pokemon_to_poke_engine_pkmn,
)

from .test_team_pool_inference import (
    ALTERNATE_SPECIES,
    DEFAULT_SPECIES,
    _battle,
    _match,
    _pokemon,
    _pool,
    _team_record,
)


_ORIGINAL_POKEMON_FORMAT = FoulPlayConfig.pokemon_format


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"


def tearDownModule():
    FoulPlayConfig.pokemon_format = _ORIGINAL_POKEMON_FORMAT


def _variant(
    team_id,
    *,
    moves=("tackle", "protect", "rest", "sleeptalk"),
    item="leftovers",
    ability=None,
    level=100,
    nature="adamant",
    evs=None,
    ivs=None,
):
    team = _team_record(team_id)
    member = team.pokemon[0]
    member = replace(
        member,
        move_ids=tuple(moves),
        item_id=item,
        base_ability_id=ability or member.base_ability_id,
        current_ability_id=ability or member.base_ability_id,
        level=level,
        nature_id=nature,
        evs=evs or member.evs,
        ivs=ivs or member.ivs,
    )
    return replace(team, pokemon=(member,) + team.pokemon[1:])


def _scenario(records=None, *, policy=None, format_id="gen9tugs"):
    records = tuple(records or (_variant("teamone"),))
    pool = _pool(format_id="gen9tugs", records=records)
    registry = TeamPoolRegistry((pool,))
    battle = _battle(format_id=format_id, policy=policy)
    _match(battle, registry)
    if battle.team_inference.baseline_candidate_count:
        active_index = next(
            index
            for index, pokemon in enumerate(battle.opponent.reserve)
            if pokemon.name == "pikachu"
        )
        battle.opponent.active = battle.opponent.reserve.pop(active_index)
    battle.user.last_selected_move = LastUsedMove("weedle", "tackle", 0)
    return battle, registry, pool


def _process(battle, *messages):
    battle.msg_list = list(messages)
    process_battle_updates(battle)


def _filtered(battle, registry):
    return filter_team_candidates(battle.team_inference, registry)


def _evidence(battle):
    return battle.team_inference.observation_ledger.member("pikachu")


class TestPolicyLedgerAndIsolation(unittest.TestCase):
    def test_01_closed_candidates_are_reference_only(self):
        battle, _, _ = _scenario()
        self.assertIs(CandidateAccess.REFERENCE_ONLY, battle.team_inference.candidate_access)
        self.assertIs(CandidateAccess.REFERENCE_ONLY, battle.team_inference.safe_summary().candidate_access)

    def test_02_open_remains_placeholder_without_record_exposure(self):
        battle, _, _ = _scenario(policy=TeamSheetPolicy.OPEN)
        context = battle.team_inference
        self.assertIs(CandidateAccess.OPEN_ELIGIBLE, context.candidate_access)
        self.assertFalse(context.exact_fields_available)
        self.assertFalse(hasattr(context, "team_records"))

    def test_03_observation_ledger_is_immutable(self):
        battle, _, _ = _scenario()
        ledger = battle.team_inference.observation_ledger
        self.assertIsInstance(ledger, OpponentObservationLedger)
        with self.assertRaises(FrozenInstanceError):
            ledger.revision = 1
        with self.assertRaises(FrozenInstanceError):
            ledger.members[0].level = 50

    def test_04_two_battles_do_not_share_observations(self):
        first, _, _ = _scenario()
        second, _, _ = _scenario()
        _process(first, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        self.assertEqual(("ironhead",), _evidence(first).selected_move_ids)
        self.assertEqual((), _evidence(second).selected_move_ids)

    def test_05_deepcopy_has_independent_ledger_and_context(self):
        battle, _, _ = _scenario()
        copied = copy.deepcopy(battle)
        _process(copied, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        self.assertIsNot(copied.team_inference, battle.team_inference)
        self.assertIsNot(copied.team_inference.observation_ledger, battle.team_inference.observation_ledger)
        self.assertEqual((), _evidence(battle).selected_move_ids)

    def test_06_registry_is_not_stored_on_battle(self):
        battle, _, _ = _scenario()
        self.assertFalse(any(isinstance(value, TeamPoolRegistry) for value in vars(battle).values()))

    def test_07_registry_is_not_stored_on_context(self):
        battle, _, _ = _scenario()
        context = battle.team_inference
        self.assertFalse(any(isinstance(getattr(context, slot), TeamPoolRegistry) for slot in context.__slots__))

    def test_08_original_roster_candidate_ids_remain_immutable(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two", moves=("ironhead", "protect", "rest", "sleeptalk"))))
        baseline = battle.team_inference.baseline_candidate_ids
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        filtered = _filtered(battle, registry)
        self.assertEqual(baseline, filtered.baseline_candidate_ids)
        self.assertIsInstance(filtered.baseline_candidate_ids, tuple)

    def test_09_active_candidates_recompute_from_baseline(self):
        records = (_variant("left", item="leftovers"), _variant("scarf", item="choicescarf"))
        battle, registry, _ = _scenario(records)
        _process(battle, "|-item|p2a: Pikachu|Leftovers")
        battle.team_inference = _filtered(battle, registry)
        self.assertEqual(1, battle.team_inference.candidate_count)
        _process(battle, "|-item|p2a: Pikachu|Choice Scarf|[from] move: Trick")
        recomputed = _filtered(battle, registry)
        self.assertEqual(2, recomputed.baseline_candidate_count)
        self.assertEqual(
            "leftovers",
            recomputed.observation_ledger.member("pikachu").initial_item_id,
        )
        self.assertEqual(1, recomputed.candidate_count)


class TestMoveEvidence(unittest.TestCase):
    def setUp(self):
        self.records = (
            _variant("tackle"),
            _variant("iron", moves=("ironhead", "protect", "rest", "sleeptalk")),
        )

    def _assert_iron_filters(self, move_line, *terminal_lines):
        battle, registry, _ = _scenario(self.records)
        _process(battle, move_line, *terminal_lines)
        filtered = _filtered(battle, registry)
        self.assertEqual(("ironhead",), _evidence(battle).selected_move_ids)
        self.assertEqual("iron", filtered.candidate_ids[0].team_record_id.team_id)

    def test_10_ordinary_selected_move_filters(self):
        self._assert_iron_filters("|move|p2a: Pikachu|Iron Head|p1a: Weedle")

    def test_11_multiple_moves_filter_cumulatively(self):
        records = (
            _variant("tackle"),
            _variant("both", moves=("tackle", "ironhead", "rest", "sleeptalk")),
            _variant("iron", moves=("ironhead", "protect", "rest", "sleeptalk")),
        )
        battle, registry, _ = _scenario(records)
        _process(battle, "|move|p2a: Pikachu|Tackle|p1a: Weedle")
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        filtered = _filtered(battle, registry)
        self.assertEqual(("ironhead", "tackle"), _evidence(battle).selected_move_ids)
        self.assertEqual("both", filtered.candidate_ids[0].team_record_id.team_id)

    def test_12_selected_move_that_misses_filters(self):
        self._assert_iron_filters("|move|p2a: Pikachu|Iron Head|p1a: Weedle|[miss]", "|-miss|p2a: Pikachu|p1a: Weedle")

    def test_13_selected_move_that_hits_immunity_filters(self):
        self._assert_iron_filters("|move|p2a: Pikachu|Iron Head|p1a: Weedle", "|-immune|p1a: Weedle")

    def test_14_selected_move_that_fails_filters(self):
        self._assert_iron_filters("|move|p2a: Pikachu|Iron Head||[still]", "|-fail|p2a: Pikachu")

    def test_15_closing_jaws_selected_iron_head_filters(self):
        records = tuple(
            _variant(
                record.record_id.team_id,
                moves=record.pokemon[0].move_ids,
                ability="closingjaws",
            )
            for record in self.records
        )
        battle, registry, _ = _scenario(records)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle|[from] ability: Closing Jaws")
        self.assertEqual("iron", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)
        self.assertIn(PublicObservationSource.CLOSING_JAWS_SELECTED_MOVE, _evidence(battle).provenance)

    def test_16_closing_jaws_failed_sucker_punch_filters(self):
        records = (
            _variant("tackle", ability="closingjaws"),
            _variant("sucker", moves=("suckerpunch", "protect", "rest", "sleeptalk"), ability="closingjaws"),
        )
        battle, registry, _ = _scenario(records)
        _process(battle, "|move|p2a: Pikachu|Sucker Punch||[from] ability: Closing Jaws|[still]", "|-fail|p2a: Pikachu")
        self.assertEqual("sucker", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_17_magic_bounce_called_move_does_not_filter(self):
        records = tuple(_variant(record.record_id.team_id, moves=record.pokemon[0].move_ids, ability="magicbounce") for record in self.records)
        battle, registry, _ = _scenario(records)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle|[from] ability: Magic Bounce")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_18_metronome_called_move_does_not_filter(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle|[from] move: Metronome")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_19_sleep_talk_called_move_does_not_filter(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle|[from] move: Sleep Talk")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_20_struggle_does_not_filter(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|move|p2a: Pikachu|Struggle|p1a: Weedle")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_21_bot_owned_move_is_ignored(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|move|p1a: Weedle|Iron Head|p2a: Pikachu")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)


class TestItemEvidence(unittest.TestCase):
    def setUp(self):
        self.records = (_variant("left", item="leftovers"), _variant("scarf", item="choicescarf"))

    def _assert_item(self, message, item_id, expected_team):
        battle, registry, _ = _scenario(self.records)
        _process(battle, message)
        self.assertEqual(item_id, _evidence(battle).initial_item_id)
        self.assertEqual(expected_team, _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_22_direct_initial_item_reveal_filters(self):
        self._assert_item("|-item|p2a: Pikachu|Leftovers", "leftovers", "left")

    def test_22a_leftovers_heal_records_item_for_healed_pokemon(self):
        self._assert_item(
            "|-heal|p2a: Pikachu|100/100|[from] item: Leftovers",
            "leftovers",
            "left",
        )

    def test_22b_black_sludge_heal_records_item_for_healed_pokemon(self):
        records = (
            _variant("sludge", item="blacksludge"),
            _variant("scarf", item="choicescarf"),
        )
        battle, registry, _ = _scenario(records)
        _process(battle, "|-heal|p2a: Pikachu|100/100|[from] item: Black Sludge")
        self.assertEqual("blacksludge", _evidence(battle).initial_item_id)
        self.assertEqual(
            "sludge",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_22c_repeated_identical_item_healing_is_idempotent(self):
        battle, _, _ = _scenario(self.records)
        message = "|-heal|p2a: Pikachu|100/100|[from] item: Leftovers"
        _process(battle, message)
        first_ledger = battle.team_inference.observation_ledger
        _process(battle, message)
        self.assertIs(first_ledger, battle.team_inference.observation_ledger)
        self.assertEqual("leftovers", _evidence(battle).initial_item_id)

    def test_22d_item_heal_uses_only_safe_public_provenance(self):
        battle, _, _ = _scenario(self.records)
        _process(battle, "|-heal|p2a: Pikachu|100/100|[from] item: Leftovers")
        self.assertEqual(
            {
                PublicObservationSource.TEAM_PREVIEW,
                PublicObservationSource.DIRECT_ITEM_REVEAL,
            },
            set(_evidence(battle).provenance),
        )

    def test_22e_move_sourced_heal_does_not_record_item(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-heal|p2a: Pikachu|100/100|[from] move: Recover")
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_22f_ability_sourced_heal_does_not_record_item(self):
        records = (
            _variant("left", item="leftovers", ability="voltabsorb"),
            _variant("scarf", item="choicescarf", ability="voltabsorb"),
        )
        battle, registry, _ = _scenario(records)
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Volt Absorb|[of] p1a: Weedle",
        )
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_22g_wish_heal_does_not_record_item(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-heal|p2a: Pikachu|100/100|[from] move: Wish")
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_22h_leech_seed_heal_does_not_record_item(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-heal|p2a: Pikachu|100/100|[from] Leech Seed|[of] p1a: Weedle")
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_22i_bare_heal_does_not_record_item(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-heal|p2a: Pikachu|100/100")
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_22j_rocky_helmet_damage_is_not_attributed_to_damaged_pokemon(self):
        battle, _, _ = _scenario(self.records)
        _process(
            battle,
            "|-damage|p2a: Pikachu|90/100|[from] item: Rocky Helmet|[of] p1a: Weedle",
        )
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertNotEqual("rockyhelmet", battle.opponent.active.item)
        self.assertEqual("rockyhelmet", battle.user.active.item)

    def test_22k_item_damage_with_of_does_not_use_damaged_target_as_owner(self):
        battle, _, _ = _scenario(self.records)
        _process(
            battle,
            "|-damage|p2a: Pikachu|90/100|[from] item: Sticky Barb|[of] p1a: Weedle",
        )
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertNotEqual("stickybarb", battle.opponent.active.item)
        self.assertEqual("stickybarb", battle.user.active.item)

    def test_23_item_activation_filters(self):
        self._assert_item("|-activate|p2a: Pikachu|item: Leftovers", "leftovers", "left")

    def test_24_consumed_item_filters(self):
        records = (_variant("sitrus", item="sitrusberry"), _variant("scarf", item="choicescarf"))
        battle, registry, _ = _scenario(records)
        _process(battle, "|-enditem|p2a: Pikachu|Sitrus Berry|[eat]")
        self.assertEqual("sitrus", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_25_knock_off_removal_filters(self):
        self._assert_item("|-enditem|p2a: Pikachu|Leftovers|[from] move: Knock Off|[of] p1a: Weedle", "leftovers", "left")

    def test_26_corrosive_gas_removal_filters(self):
        self._assert_item("|-enditem|p2a: Pikachu|Leftovers|[from] move: Corrosive Gas", "leftovers", "left")

    def test_27_trick_does_not_mistake_received_item_for_initial(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-item|p2a: Pikachu|Choice Scarf|[from] move: Trick")
        self.assertTrue(_evidence(battle).item_ambiguous)
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_28_ambiguous_item_history_does_not_overfilter(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-item|p2a: Pikachu|Leftovers")
        _process(battle, "|-item|p2a: Pikachu|Choice Scarf|[from] move: Switcheroo")
        evidence = _evidence(battle)
        self.assertEqual("leftovers", evidence.initial_item_id)
        self.assertFalse(evidence.item_ambiguous)
        self.assertEqual(
            {PublicObservationSource.TEAM_PREVIEW,
             PublicObservationSource.DIRECT_ITEM_REVEAL,
             PublicObservationSource.ITEM_ACQUIRED},
            set(evidence.provenance),
        )
        self.assertEqual(1, _filtered(battle, registry).candidate_count)

    def test_28a_repeated_acquisition_preserves_original_item_idempotently(self):
        battle, registry, _ = _scenario(self.records)
        _process(battle, "|-item|p2a: Pikachu|Leftovers")
        message = "|-item|p2a: Pikachu|Choice Scarf|[from] move: Trick"
        _process(battle, message)
        first = battle.team_inference.observation_ledger
        _process(battle, message)
        self.assertIs(first, battle.team_inference.observation_ledger)
        self.assertEqual("leftovers", _evidence(battle).initial_item_id)
        self.assertEqual(1, _filtered(battle, registry).candidate_count)

    def test_29_item_none_alone_does_not_mean_submitted_itemless(self):
        battle, registry, _ = _scenario(self.records)
        battle.opponent.active.item = None
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_30_inferred_item_is_ignored(self):
        battle, registry, _ = _scenario(self.records)
        battle.opponent.active.item = "leftovers"
        battle.opponent.active.item_inferred = True
        self.assertIsNone(_evidence(battle).initial_item_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)


class TestAbilityEvidence(unittest.TestCase):
    def _ability_scenario(self, first, second="static"):
        return _scenario((_variant("first", ability=first), _variant("second", ability=second)))

    def test_31a_heal_opponent_actor_owns_ability_despite_user_source(self):
        battle, registry, _ = self._ability_scenario("waterabsorb", "static")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb|[of] p1a: Weedle",
        )
        self.assertEqual("waterabsorb", _evidence(battle).base_ability_id)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31b_heal_user_actor_does_not_attribute_ability_to_opponent_source(self):
        battle, registry, _ = self._ability_scenario("torrent", "static")
        _process(
            battle,
            "|-heal|p1a: Weedle|100/100|[from] ability: Water Absorb|[of] p2a: Pikachu",
        )
        self.assertIsNone(_evidence(battle).base_ability_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_31c_heal_opponent_actor_without_source_records_ability(self):
        battle, registry, _ = self._ability_scenario("waterabsorb", "static")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb",
        )
        self.assertEqual("waterabsorb", _evidence(battle).base_ability_id)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31d_heal_user_actor_without_source_records_no_opponent_ability(self):
        battle, registry, _ = self._ability_scenario("torrent", "static")
        _process(
            battle,
            "|-heal|p1a: Weedle|100/100|[from] ability: Water Absorb",
        )
        self.assertIsNone(_evidence(battle).base_ability_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_31e_heal_normalizes_water_absorb(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb",
        )
        self.assertEqual("waterabsorb", _evidence(battle).base_ability_id)

    def test_31f_heal_handles_arbitrary_explicit_ability_without_special_case(self):
        battle, registry, _ = self._ability_scenario("voltabsorb", "static")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Volt Absorb|[of] p1a: Weedle",
        )
        self.assertEqual("voltabsorb", _evidence(battle).base_ability_id)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31g_repeated_identical_heal_ability_evidence_is_idempotent(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        message = "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb"
        _process(battle, message)
        first_ledger = battle.team_inference.observation_ledger
        _process(battle, message)
        self.assertIs(first_ledger, battle.team_inference.observation_ledger)

    def test_31h_heal_ability_uses_safe_direct_reveal_provenance(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb",
        )
        self.assertEqual(
            {
                PublicObservationSource.TEAM_PREVIEW,
                PublicObservationSource.DIRECT_ABILITY_REVEAL,
            },
            set(_evidence(battle).provenance),
        )

    def test_31i_heal_without_explicit_ability_records_no_ability(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(battle, "|-heal|p2a: Pikachu|100/100")
        self.assertIsNone(_evidence(battle).base_ability_id)

    def test_31j_item_heal_records_only_item_evidence(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] item: Leftovers",
        )
        self.assertEqual("leftovers", _evidence(battle).initial_item_id)
        self.assertIsNone(_evidence(battle).base_ability_id)

    def test_31k_move_heal_does_not_fabricate_ability(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] move: Recover",
        )
        self.assertIsNone(_evidence(battle).base_ability_id)

    def test_31l_heal_does_not_take_ability_name_from_source_annotation(self):
        battle, _, _ = self._ability_scenario("waterabsorb")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] move: Recover|[of] p1a: Water Absorb",
        )
        self.assertIsNone(_evidence(battle).base_ability_id)

    def test_31m_exact_defect_direction_does_not_assign_water_absorb_to_source(self):
        battle, _, _ = self._ability_scenario("torrent", "static")
        _process(
            battle,
            "|-heal|p1a: Jellicent|100/100|[from] ability: Water Absorb|[of] p2a: Swampert",
        )
        self.assertIsNone(_evidence(battle).base_ability_id)

    def test_31n_inverse_heal_direction_assigns_water_absorb_to_opponent_actor(self):
        battle, _, _ = self._ability_scenario("waterabsorb", "static")
        _process(
            battle,
            "|-heal|p2a: Pikachu|100/100|[from] ability: Water Absorb|[of] p1a: Weedle",
        )
        self.assertEqual("waterabsorb", _evidence(battle).base_ability_id)

    def test_31o_iron_barbs_damage_remains_owned_by_opponent_source(self):
        battle, registry, _ = self._ability_scenario("ironbarbs", "static")
        _process(
            battle,
            "|-damage|p1a: Weedle|90/100|[from] ability: Iron Barbs|[of] p2a: Pikachu",
        )
        self.assertEqual("ironbarbs", _evidence(battle).base_ability_id)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31p_rough_skin_damage_remains_owned_by_opponent_source(self):
        battle, registry, _ = self._ability_scenario("roughskin", "static")
        _process(
            battle,
            "|-damage|p1a: Weedle|90/100|[from] ability: Rough Skin|[of] p2a: Pikachu",
        )
        self.assertEqual("roughskin", _evidence(battle).base_ability_id)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31q_damaged_opponent_target_does_not_receive_source_ability(self):
        for ability in ("Iron Barbs", "Rough Skin"):
            with self.subTest(ability=ability):
                battle, _, _ = self._ability_scenario("torrent", "static")
                _process(
                    battle,
                    f"|-damage|p2a: Pikachu|90/100|[from] ability: {ability}|[of] p1a: Weedle",
                )
                self.assertIsNone(_evidence(battle).base_ability_id)
                self.assertNotEqual(normalize_name(ability), battle.opponent.active.ability)

    def test_31r_trace_ownership_remains_on_tracing_opponent_base_ability(self):
        battle, registry, _ = self._ability_scenario("trace", "closingjaws")
        _process(
            battle,
            "|-ability|p2a: Pikachu|Closing Jaws|Trace|[from] ability: Trace|[of] p1a: Weedle",
        )
        self.assertEqual("trace", _evidence(battle).base_ability_id)
        self.assertTrue(_evidence(battle).current_ability_changed)
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31s_inverse_trace_attribution_to_opponent_source_remains_unchanged(self):
        battle, registry, _ = self._ability_scenario("static", "lightningrod")
        _process(
            battle,
            "|-ability|p1a: Weedle|Static|Trace|[from] ability: Trace|[of] p2a: Pikachu",
        )
        self.assertEqual("static", _evidence(battle).base_ability_id)
        self.assertIn(
            PublicObservationSource.TRACE_BASE_ABILITY,
            _evidence(battle).provenance,
        )
        self.assertEqual(
            "first",
            _filtered(battle, registry).candidate_ids[0].team_record_id.team_id,
        )

    def test_31t_neutralizing_gas_suppression_does_not_overwrite_base_ability(self):
        battle, _, _ = self._ability_scenario("static", "neutralizinggas")
        _process(battle, "|-ability|p2a: Pikachu|Static")
        _process(battle, "|-ability|p1a: Weedle|Neutralizing Gas")
        self.assertEqual("static", _evidence(battle).base_ability_id)
        self.assertFalse(_evidence(battle).conflicting_public_evidence)

    def test_31_normal_base_ability_reveal_filters(self):
        battle, registry, _ = self._ability_scenario("static", "lightningrod")
        _process(battle, "|-ability|p2a: Pikachu|Static")
        self.assertEqual("static", _evidence(battle).base_ability_id)
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_32_download_reveal_filters(self):
        battle, registry, _ = self._ability_scenario("download", "static")
        _process(battle, "|-ability|p2a: Pikachu|Download")
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_33_closing_jaws_ability_reveal_filters(self):
        battle, registry, _ = self._ability_scenario("closingjaws", "static")
        _process(battle, "|-activate|p2a: Pikachu|ability: Closing Jaws")
        self.assertEqual("closingjaws", _evidence(battle).base_ability_id)
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_34_neutralizing_gas_reveal_filters(self):
        battle, registry, _ = self._ability_scenario("neutralizinggas", "static")
        _process(battle, "|-ability|p2a: Pikachu|Neutralizing Gas")
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_35_trace_records_trace_as_base_ability(self):
        battle, registry, _ = self._ability_scenario("trace", "closingjaws")
        _process(battle, "|-ability|p2a: Pikachu|Closing Jaws|Trace|[from] ability: Trace|[of] p1a: Weedle")
        self.assertEqual("trace", _evidence(battle).base_ability_id)
        self.assertTrue(_evidence(battle).current_ability_changed)
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_36_trace_copied_current_ability_does_not_replace_trace(self):
        battle, registry, _ = self._ability_scenario("trace", "closingjaws")
        _process(battle, "|-ability|p2a: Pikachu|Closing Jaws|Trace|[from] ability: Trace|[of] p1a: Weedle")
        _process(battle, "|-activate|p2a: Pikachu|ability: Closing Jaws")
        self.assertEqual("trace", _evidence(battle).base_ability_id)
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_37_gastro_acid_does_not_erase_base_ability(self):
        battle, registry, _ = self._ability_scenario("static", "lightningrod")
        _process(battle, "|-ability|p2a: Pikachu|Static")
        _process(battle, "|-start|p2a: Pikachu|Gastro Acid")
        self.assertEqual("static", _evidence(battle).base_ability_id)
        self.assertEqual("first", _filtered(battle, registry).candidate_ids[0].team_record_id.team_id)

    def test_38_inferred_ability_is_ignored(self):
        battle, registry, _ = self._ability_scenario("static", "lightningrod")
        battle.opponent.active.ability = "static"
        self.assertIsNone(_evidence(battle).base_ability_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_39_impossible_abilities_are_ignored(self):
        battle, registry, _ = self._ability_scenario("static", "lightningrod")
        battle.opponent.active.impossible_abilities.add("lightningrod")
        self.assertIsNone(_evidence(battle).base_ability_id)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)


class TestFilteringRulesAndStates(unittest.TestCase):
    def test_40_exact_level_filters(self):
        battle, registry, _ = _scenario((_variant("hundred", level=100), _variant("fifty", level=50)))
        filtered = _filtered(battle, registry)
        self.assertEqual("hundred", filtered.candidate_ids[0].team_record_id.team_id)
        self.assertIs(CandidateFilterState.REDUCED, filtered.filter_state)

    def test_41_exact_form_remains_significant(self):
        pool = _pool(records=(_team_record("ordinary"),))
        registry = TeamPoolRegistry((pool,))
        battle = _battle(species=ALTERNATE_SPECIES)
        _match(battle, registry)
        self.assertIs(TeamPoolMatchState.NO_MATCH, battle.team_inference.match_state)
        self.assertIs(CandidateFilterState.NOT_APPLICABLE, _filtered(battle, registry).filter_state)

    def test_42_preview_order_remains_irrelevant(self):
        pool = _pool(records=(_team_record("ordinary"),))
        registry = TeamPoolRegistry((pool,))
        first = _battle()
        second = _battle(species=tuple(reversed(DEFAULT_SPECIES)))
        _match(first, registry)
        _match(second, registry)
        self.assertEqual(first.team_inference.baseline_candidate_ids, second.team_inference.baseline_candidate_ids)
        self.assertEqual(_filtered(first, registry).candidate_ids, _filtered(second, registry).candidate_ids)

    def test_43_nature_and_ev_differences_do_not_filter(self):
        records = (
            _variant("adamant", nature="adamant", evs=StatValues(4, 252, 0, 0, 0, 252)),
            _variant("modest", nature="modest", evs=StatValues(252, 0, 0, 252, 4, 0)),
        )
        battle, registry, _ = _scenario(records)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_44_iv_differences_do_not_filter(self):
        records = (
            _variant("full", ivs=StatValues(31, 31, 31, 31, 31, 31)),
            _variant("zero", ivs=StatValues(0, 0, 0, 0, 0, 0)),
        )
        battle, registry, _ = _scenario(records)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_45_speed_range_evidence_is_ignored(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two")))
        battle.opponent.active.speed_range = (123, 124)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_46_reverse_damage_evidence_is_ignored(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two")))
        battle.opponent.active.stats[constants.ATTACK] = 999
        battle.opponent.active.hp = 1
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_47_candidate_ordering_remains_deterministic(self):
        records = (_variant("zeta"), _variant("alpha"), _variant("middle"))
        battle, registry, _ = _scenario(records)
        first = _filtered(battle, registry).candidate_ids
        second = _filtered(battle, registry).candidate_ids
        self.assertEqual(tuple(sorted(first)), first)
        self.assertEqual(first, second)

    def test_48_multi_pool_candidates_filter_independently(self):
        first_pool = _pool("firstpool", records=(_variant("left", moves=("tackle", "protect", "rest", "sleeptalk")),))
        second_pool = _pool("secondpool", records=(_variant("iron", moves=("ironhead", "protect", "rest", "sleeptalk")),))
        registry = TeamPoolRegistry((first_pool, second_pool))
        battle = _battle()
        _match(battle, registry)
        battle.opponent.active = battle.opponent.reserve.pop(0)
        battle.user.last_selected_move = LastUsedMove("weedle", "tackle", 0)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        filtered = _filtered(battle, registry)
        self.assertEqual(2, filtered.baseline_candidate_count)
        self.assertEqual("secondpool", filtered.candidate_ids[0].pool_identity.pool_id)

    def test_49_unique_closed_candidate_remains_reference_only(self):
        battle, registry, _ = _scenario()
        filtered = _filtered(battle, registry)
        self.assertEqual(1, filtered.candidate_count)
        self.assertIs(CandidateAccess.REFERENCE_ONLY, filtered.candidate_access)

    def test_50_unique_closed_candidate_does_not_populate_hidden_state(self):
        battle, registry, _ = _scenario()
        before = (battle.opponent.active.item, battle.opponent.active.ability, tuple(battle.opponent.active.moves), battle.opponent.active.nature, tuple(battle.opponent.active.evs))
        _filtered(battle, registry)
        after = (battle.opponent.active.item, battle.opponent.active.ability, tuple(battle.opponent.active.moves), battle.opponent.active.nature, tuple(battle.opponent.active.evs))
        self.assertEqual(before, after)

    def test_51_multiple_compatible_candidates_remain_ambiguous(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two")))
        filtered = _filtered(battle, registry)
        self.assertIs(TeamPoolMatchState.AMBIGUOUS_MATCH, filtered.match_state)
        self.assertEqual(2, filtered.candidate_count)

    def test_52_some_eliminated_produces_reduced(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two", moves=("ironhead", "protect", "rest", "sleeptalk"))))
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        self.assertIs(CandidateFilterState.REDUCED, _filtered(battle, registry).filter_state)

    def test_53_no_eliminated_produces_consistent(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two")))
        self.assertIs(CandidateFilterState.CONSISTENT, _filtered(battle, registry).filter_state)

    def test_54_all_eliminated_produces_exhausted(self):
        battle, registry, _ = _scenario((_variant("one"), _variant("two")))
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        filtered = _filtered(battle, registry)
        self.assertIs(CandidateFilterState.EXHAUSTED, filtered.filter_state)
        self.assertEqual(0, filtered.candidate_count)

    def test_55_exhaustion_keeps_generic_fallback_eligible(self):
        battle, registry, _ = _scenario()
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        self.assertTrue(_filtered(battle, registry).generic_fallback_eligible)

    def test_56_exhaustion_does_not_insert_generic_ids(self):
        battle, registry, _ = _scenario()
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        filtered = _filtered(battle, registry)
        self.assertEqual((), filtered.candidate_ids)
        self.assertEqual(1, filtered.baseline_candidate_count)

    def test_57_filtering_no_match_is_safe_noop(self):
        pool = _pool(records=(_team_record(),))
        registry = TeamPoolRegistry((pool,))
        battle = _battle(species=ALTERNATE_SPECIES)
        _match(battle, registry)
        filtered = _filtered(battle, registry)
        self.assertIs(CandidateFilterState.NOT_APPLICABLE, filtered.filter_state)

    def test_58_filtering_incomplete_preview_is_safe_noop(self):
        pool = _pool(records=(_team_record(),))
        registry = TeamPoolRegistry((pool,))
        battle = _battle()
        battle.opponent.reserve.pop()
        _match(battle, registry)
        filtered = _filtered(battle, registry)
        self.assertIs(TeamPoolMatchState.INCOMPLETE_PREVIEW, filtered.match_state)
        self.assertIs(CandidateFilterState.NOT_APPLICABLE, filtered.filter_state)

    def test_conflicting_public_evidence_has_distinct_state(self):
        battle, registry, _ = _scenario((_variant("left", item="leftovers"), _variant("scarf", item="choicescarf")))
        context = battle.team_inference
        context.record_initial_item("pikachu", "leftovers", PublicObservationSource.DIRECT_ITEM_REVEAL)
        context.record_initial_item("pikachu", "choicescarf", PublicObservationSource.ITEM_ACTIVATION)
        filtered = _filtered(battle, registry)
        self.assertIs(CandidateFilterState.CONFLICTING_PUBLIC_EVIDENCE, filtered.filter_state)
        self.assertEqual(2, filtered.candidate_count)


class TestMutationAndScopeBoundaries(unittest.TestCase):
    def test_59_filtering_creates_no_cache_or_file(self):
        battle, registry, _ = _scenario()
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected file access")):
            filtered = _filtered(battle, registry)
        self.assertEqual(1, filtered.candidate_count)

    def test_60_filtering_performs_no_network_activity(self):
        battle, registry, _ = _scenario()
        with patch.object(socket, "create_connection", side_effect=AssertionError("unexpected network access")):
            filtered = _filtered(battle, registry)
        self.assertEqual(1, filtered.candidate_count)

    def test_61_filtering_does_not_mutate_pool_records(self):
        battle, registry, pool = _scenario((_variant("one"), _variant("two")))
        teams_before = pool.teams
        index_before = tuple(pool.roster_index.items())
        repr_before = repr(pool)
        _filtered(battle, registry)
        self.assertIs(teams_before, pool.teams)
        self.assertEqual(index_before, tuple(pool.roster_index.items()))
        self.assertEqual(repr_before, repr(pool))

    def test_62_filtering_does_not_mutate_global_move_data(self):
        battle, registry, _ = _scenario()
        before = copy.deepcopy(all_move_json)
        _filtered(battle, registry)
        self.assertEqual(before, all_move_json)

    def test_63_filtering_does_not_mutate_global_pokedex_data(self):
        battle, registry, _ = _scenario()
        before = copy.deepcopy(pokedex)
        _filtered(battle, registry)
        self.assertEqual(before, pokedex)

    def test_64_filtering_does_not_mutate_team_datasets(self):
        battle, registry, _ = _scenario()
        datasets = TeamDatasets()
        before = dict(vars(datasets))
        _filtered(battle, registry)
        self.assertEqual(before, vars(datasets))

    def test_65_filtering_does_not_mutate_smogon_sets(self):
        battle, registry, _ = _scenario()
        sets = SmogonSets()
        before = dict(vars(sets))
        _filtered(battle, registry)
        self.assertEqual(before, vars(sets))

    def test_66_own_team_state_remains_unchanged(self):
        battle, registry, _ = _scenario()
        _process(battle, "|move|p1a: Weedle|Iron Head|p2a: Pikachu")
        active = battle.user.active
        reserve = tuple(battle.user.reserve)
        snapshot = (active.item, active.ability, tuple(active.moves), active.nature, tuple(active.evs))
        _filtered(battle, registry)
        self.assertIs(active, battle.user.active)
        self.assertEqual(reserve, tuple(battle.user.reserve))
        self.assertEqual(snapshot, (active.item, active.ability, tuple(active.moves), active.nature, tuple(active.evs)))

    def test_67_no_search_source_contains_team_pool_resolution(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ("fp/search/main.py", "fp/search/standard_battles.py", "fp/search/helpers.py"):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("TeamPoolRegistry", source)
            self.assertNotIn("filter_team_candidates", source)
            self.assertNotIn("baseline_candidate_ids", source)

    def test_68_no_poke_engine_serialization_schema_change(self):
        battle, registry, _ = _scenario()
        battle.team_inference = _filtered(battle, registry)
        state = battle_to_poke_engine_state(battle)
        self.assertFalse(hasattr(state, "team_inference"))
        self.assertNotIn("team_inference", state.to_string())

    def test_69_preview_behavior_remains_functional_without_registry(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.user.active = _pokemon("weedle")
        battle.initialize_team_preview([f"{species}, L100" for species in DEFAULT_SPECIES], "gen9tugs")
        StandardBattleMode.match_team_preview(battle, None)
        self.assertEqual(6, len(battle.opponent.reserve))
        self.assertEqual(6, len(battle.team_inference.observation_ledger.members))
        self.assertIs(TeamPoolMatchState.NO_POOL, battle.team_inference.match_state)

    def test_70_phase_one_public_types_remain_frozen(self):
        record = _variant("one").pokemon[0]
        with self.assertRaises(FrozenInstanceError):
            record.item_id = "choicescarf"

    def test_71_phase_two_matching_remains_functional(self):
        battle, _, _ = _scenario()
        self.assertIs(TeamPoolMatchState.MATCHED, battle.team_inference.match_state)
        self.assertEqual(1, battle.team_inference.baseline_candidate_count)
        self.assertFalse(battle.team_inference.exact_fields_available)


class TestExistingTugsRegressions(unittest.TestCase):
    def test_72_crag_mend_behavior_remains_passing(self):
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

    def test_73_trick_room_persistent_behavior_remains_passing(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("lapras")
        battle.user.active.ability = "persistent"
        battle.opponent.active = _pokemon("pikachu")
        fieldstart(battle, ["", "-fieldstart", "move: Trick Room", "[of] p1a: Lapras", "[persistent]"])
        self.assertEqual(8, battle.trick_room_turns_remaining)

    def test_74_ancient_shell_serialization_remains_passing(self):
        pokemon = _pokemon("lapras")
        pokemon.ability = "ancientshell"
        self.assertEqual("ancientshell", pokemon_to_poke_engine_pkmn(pokemon).ability)

    def test_75_corrosive_gas_parsing_remains_passing(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("pikachu")
        battle.opponent.active = _pokemon("dustox")
        battle.opponent.active.item = "leftovers"
        remove_item(battle, ["", "-enditem", "p2a: Dustox", "Leftovers", "[from] move: Corrosive Gas"])
        self.assertIsNone(battle.opponent.active.item)
        self.assertEqual("leftovers", battle.opponent.active.removed_item)

    def test_76_closing_jaws_bookkeeping_remains_passing(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("weedle")
        battle.opponent.active = _pokemon("mawile")
        activate(battle, ["", "-activate", "p2a: Mawile", "ability: Closing Jaws"])
        self.assertEqual("closingjaws", battle.opponent.active.ability)


class TestAdditionalIsolationCases(unittest.TestCase):
    def test_tugs_pool_does_not_affect_ordinary_gen9(self):
        battle, registry, _ = _scenario(format_id="gen9ou")
        self.assertIs(TeamPoolMatchState.NO_POOL, battle.team_inference.match_state)
        self.assertIs(CandidateFilterState.NOT_APPLICABLE, _filtered(battle, registry).filter_state)

    def test_different_registries_do_not_contaminate_filtering(self):
        battle, first_registry, _ = _scenario((_variant("left", item="leftovers"),))
        other_pool = _pool("otherpool", records=(_variant("scarf", item="choicescarf"),))
        other_registry = TeamPoolRegistry((other_pool,))
        _process(battle, "|-item|p2a: Pikachu|Leftovers")
        self.assertEqual(1, _filtered(battle, first_registry).candidate_count)
        self.assertEqual(0, _filtered(battle, other_registry).candidate_count)

    def test_sequential_battles_do_not_retain_observations(self):
        first, _, _ = _scenario()
        second, _, _ = _scenario()
        _process(first, "|-item|p2a: Pikachu|Leftovers")
        self.assertEqual("leftovers", _evidence(first).initial_item_id)
        self.assertIsNone(_evidence(second).initial_item_id)

    def test_transformed_runtime_move_is_not_submitted_move_evidence(self):
        records = (_variant("one"), _variant("two", moves=("ironhead", "protect", "rest", "sleeptalk")))
        battle, registry, _ = _scenario(records)
        battle.opponent.active.volatile_statuses.append(constants.TRANSFORM)
        _process(battle, "|move|p2a: Pikachu|Iron Head|p1a: Weedle")
        self.assertEqual((), _evidence(battle).selected_move_ids)
        self.assertEqual(2, _filtered(battle, registry).candidate_count)

    def test_context_contains_no_raw_pool_or_record_objects(self):
        battle, _, _ = _scenario()
        forbidden = (TeamPoolRegistry, TeamPool, TeamRecord, PokemonRecord)
        self.assertFalse(any(isinstance(getattr(battle.team_inference, slot), forbidden) for slot in battle.team_inference.__slots__))


if __name__ == "__main__":
    unittest.main()
