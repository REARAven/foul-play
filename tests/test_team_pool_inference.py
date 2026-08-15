import copy
import unittest
from copy import deepcopy
from unittest.mock import patch

from fp import constants
from fp.battle.helpers import normalize_name
from fp.battle.protocol import activate, fieldstart, remove_item, unlikely_to_have_choice_item
from fp.battle.state import Battle, Pokemon
from fp.battle.team_inference import (
    ObservationFilterState,
    TeamInferenceSummary,
    TeamPoolCandidateId,
    TeamPoolMatchState,
    TeamSheetPolicy,
)
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.team_pools import (
    PokemonRecord,
    PoolIdentity,
    SourceLocation,
    StatValues,
    TeamPool,
    TeamPoolRegistry,
    TeamRecord,
    TeamRecordId,
    canonical_roster_key,
)
from fp.format_spec import FormatSpec
from fp.modes.standard_battle import StandardBattleMode
from fp.search.poke_engine_helpers import (
    battle_to_poke_engine_state,
    pokemon_to_poke_engine_pkmn,
)


DEFAULT_SPECIES = (
    "pikachu",
    "charizard",
    "blastoise",
    "venusaur",
    "snorlax",
    "gengar",
)
ALTERNATE_SPECIES = (
    "raichu",
    "charizard",
    "blastoise",
    "venusaur",
    "snorlax",
    "gengar",
)
HIDDEN_MOVES = ("tackle", "protect", "rest", "sleeptalk")
TEST_EVS = (0, 0, 0, 0, 0, 0)
_ORIGINAL_POKEMON_FORMAT = FoulPlayConfig.pokemon_format


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"


def tearDownModule():
    FoulPlayConfig.pokemon_format = _ORIGINAL_POKEMON_FORMAT


def _pokemon(species_id, level=100):
    return Pokemon(species_id, level, evs=TEST_EVS)


def _base_species_id(species_id):
    return normalize_name(pokedex[species_id].get("baseSpecies", species_id))


def _ability_id(species_id):
    return normalize_name(next(iter(pokedex[species_id]["abilities"].values())))


def _pokemon_record(species_id, index):
    ability_id = _ability_id(species_id)
    return PokemonRecord(
        slot_id=f"slot{index}",
        species_id=species_id,
        base_species_id=_base_species_id(species_id),
        item_id="leftovers",
        base_ability_id=ability_id,
        current_ability_id=ability_id,
        move_ids=HIDDEN_MOVES,
        nature_id="adamant",
        evs=StatValues(4, 252, 0, 0, 0, 252),
        ivs=StatValues(31, 31, 31, 31, 31, 31),
        level=100,
        public_fields=("species_id",),
        metadata={"synthetic": True},
        source_location=SourceLocation("<synthetic>", f"$.pokemon[{index - 1}]"),
    )


def _team_record(team_id="teamone", variant_id="default", species=DEFAULT_SPECIES):
    pokemon = tuple(
        _pokemon_record(species_id, index)
        for index, species_id in enumerate(species, start=1)
    )
    return TeamRecord(
        record_id=TeamRecordId(team_id, variant_id),
        variant_of=None,
        display_name="Synthetic team",
        pokemon=pokemon,
        roster_key=canonical_roster_key(species),
        metadata={"synthetic": True},
        source_location=SourceLocation("<synthetic>", "$.teams[0]"),
    )


def _pool(
    pool_id="syntheticpool",
    pool_version="1",
    format_id="gen9tugs",
    records=None,
):
    if records is None:
        records = (_team_record(),)
    return TeamPool(
        schema_version=1,
        identity=PoolIdentity(pool_id, pool_version, format_id),
        patch_version="syntheticpatch",
        source_documents=("syntheticsource",),
        display_name="Synthetic pool",
        default_public_fields=("species_id",),
        metadata={"synthetic": True},
        teams=tuple(records),
    )


def _preview(species=DEFAULT_SPECIES):
    return [_pokemon(species_id) for species_id in species]


def _battle(format_id="gen9tugs", policy=None, species=DEFAULT_SPECIES):
    battle = Battle("synthetic-battle", team_sheet_policy=policy)
    battle.pokemon_format = format_id
    battle.generation = FormatSpec.from_format_string(format_id).generation
    battle.battle_type = battle.format_spec.battle_type
    battle.mode = StandardBattleMode()
    battle.user.name = "p1"
    battle.opponent.name = "p2"
    battle.user.active = _pokemon("weedle")
    battle.user.reserve = [_pokemon("caterpie")]
    battle.opponent.reserve = _preview(species)
    return battle


def _match(battle, registry):
    StandardBattleMode.match_team_preview(battle, registry)
    return battle.team_inference


def _preview_message(species=DEFAULT_SPECIES):
    lines = ["|clearpoke"]
    lines.extend(f"|poke|p2|{species_id}, L100" for species_id in species)
    return "\n".join(lines)


async def _initialize_synthetic_user(_websocket, battle):
    own_species = ("weedle", "caterpie", "metapod", "butterfree", "kakuna", "beedrill")
    battle.user.active = _pokemon(own_species[0])
    battle.user.reserve = [_pokemon(species_id) for species_id in own_species[1:]]
    battle.rqid = 1


class _PreviewLifecycleMode(StandardBattleMode):
    def __init__(self):
        super().__init__()
        self.datasets_initialized = False
        self.preview_handled = False

    async def start_battle_common(
        self, _websocket, pokemon_battle_type, team_sheet_policy=None
    ):
        battle = Battle("synthetic-battle", team_sheet_policy=team_sheet_policy)
        battle.pokemon_format = pokemon_battle_type
        battle.generation = battle.format_spec.generation
        battle.battle_type = battle.format_spec.battle_type
        battle.mode = self
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        return battle, _preview_message()

    def initialize_team_preview_datasets(self, _format, _names, _message):
        self.datasets_initialized = True

    async def handle_team_preview(self, _battle, _websocket):
        self.preview_handled = True


class TestTeamPoolInference(unittest.TestCase):
    def setUp(self):
        self.pool = _pool()
        self.registry = TeamPoolRegistry((self.pool,))

    def test_01_default_policy_is_closed(self):
        self.assertEqual(("CLOSED", "OPEN"), tuple(TeamSheetPolicy.__members__))
        self.assertIs(TeamSheetPolicy.CLOSED, Battle(None).team_inference.policy)

    def test_02_open_can_be_explicitly_selected(self):
        battle = Battle(None, team_sheet_policy=TeamSheetPolicy.OPEN)
        self.assertIs(TeamSheetPolicy.OPEN, battle.team_inference.policy)

    def test_03_new_battle_receives_fresh_context(self):
        first = Battle("first")
        second = Battle("second")
        self.assertIsNot(first.team_inference, second.team_inference)
        self.assertIs(TeamPoolMatchState.UNINITIALIZED, first.team_inference.match_state)

    def test_04_two_battles_do_not_share_context_state(self):
        first = _battle()
        second = _battle()
        _match(first, self.registry)
        self.assertIs(TeamPoolMatchState.MATCHED, first.team_inference.match_state)
        self.assertIs(TeamPoolMatchState.UNINITIALIZED, second.team_inference.match_state)

    def test_05_deepcopy_produces_independent_candidate_state(self):
        battle = _battle()
        _match(battle, self.registry)
        copied = deepcopy(battle)
        self.assertIsNot(copied.team_inference, battle.team_inference)
        _match(copied, TeamPoolRegistry((_pool(records=(_team_record(species=ALTERNATE_SPECIES),)),)))
        self.assertIs(TeamPoolMatchState.NO_MATCH, copied.team_inference.match_state)
        self.assertIs(TeamPoolMatchState.MATCHED, battle.team_inference.match_state)
        self.assertEqual(1, battle.team_inference.candidate_count)

    def test_06_immutable_registry_may_be_shared_safely(self):
        first = _battle()
        second = _battle()
        _match(first, self.registry)
        _match(second, self.registry)
        self.assertEqual(first.team_inference.candidate_ids, second.team_inference.candidate_ids)
        self.assertIs(first.team_inference.candidate_ids[0].pool_identity, self.pool.identity)

    def test_07_no_registry_produces_no_pool_and_fallback(self):
        context = _match(_battle(), None)
        self.assertIs(TeamPoolMatchState.NO_POOL, context.match_state)
        self.assertTrue(context.generic_fallback_eligible)
        self.assertEqual((), context.candidate_ids)

    def test_08_compatible_registry_unique_roster_produces_one_candidate(self):
        context = _match(_battle(), self.registry)
        self.assertIs(TeamPoolMatchState.MATCHED, context.match_state)
        self.assertEqual(1, context.candidate_count)
        self.assertEqual(
            TeamPoolCandidateId(self.pool.identity, TeamRecordId("teamone", "default")),
            context.candidate_ids[0],
        )

    def test_09_unique_closed_match_does_not_populate_hidden_fields(self):
        battle = _battle()
        before = [
            (pkmn.item, pkmn.ability, tuple(pkmn.moves), pkmn.nature, tuple(pkmn.evs))
            for pkmn in battle.opponent.reserve
        ]
        _match(battle, self.registry)
        after = [
            (pkmn.item, pkmn.ability, tuple(pkmn.moves), pkmn.nature, tuple(pkmn.evs))
            for pkmn in battle.opponent.reserve
        ]
        self.assertEqual(before, after)

    def test_10_unique_closed_match_public_api_exposes_no_pool_records(self):
        context = _match(_battle(), self.registry)
        summary = context.safe_summary()
        self.assertIsInstance(summary, TeamInferenceSummary)
        self.assertFalse(hasattr(context, "registry"))
        self.assertFalse(hasattr(context, "team_records"))
        self.assertFalse(hasattr(context, "pokemon_records"))
        for slot_name in context.__slots__:
            self.assertNotIsInstance(
                getattr(context, slot_name),
                (TeamPoolRegistry, TeamPool, TeamRecord, PokemonRecord),
            )

    def test_11_policy_remains_closed_after_unique_match(self):
        context = _match(_battle(), self.registry)
        self.assertIs(TeamSheetPolicy.CLOSED, context.policy)
        self.assertTrue(context.generic_fallback_eligible)
        self.assertFalse(context.exact_fields_available)
        self.assertFalse(context.safe_summary().exact_fields_available)

    def test_12_shared_roster_retains_multiple_candidate_ids(self):
        records = (
            _team_record("teamtwo", "alternate"),
            _team_record("teamone", "default"),
        )
        pool = _pool(records=records)
        context = _match(_battle(), TeamPoolRegistry((pool,)))
        self.assertIs(TeamPoolMatchState.AMBIGUOUS_MATCH, context.match_state)
        self.assertEqual(
            (TeamRecordId("teamone", "default"), TeamRecordId("teamtwo", "alternate")),
            tuple(candidate.team_record_id for candidate in context.candidate_ids),
        )

    def test_13_matches_across_two_pools_are_pool_qualified(self):
        other_pool = _pool("anotherpool", records=(_team_record("otherteam"),))
        context = _match(_battle(), TeamPoolRegistry((self.pool, other_pool)))
        self.assertEqual(2, context.candidate_count)
        self.assertEqual(
            (other_pool.identity, self.pool.identity),
            tuple(candidate.pool_identity for candidate in context.candidate_ids),
        )

    def test_14_candidate_order_is_deterministic(self):
        z_pool = _pool("zpool", records=(_team_record("zteam"), _team_record("ateam")))
        a_pool = _pool("apool", records=(_team_record("mteam"),))
        registry = TeamPoolRegistry((z_pool, a_pool))
        first = _match(_battle(), registry).candidate_ids
        second = _match(_battle(species=tuple(reversed(DEFAULT_SPECIES))), registry).candidate_ids
        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(first)), first)

    def test_15_unknown_roster_produces_no_match_and_fallback(self):
        context = _match(_battle(species=ALTERNATE_SPECIES), self.registry)
        self.assertIs(TeamPoolMatchState.NO_MATCH, context.match_state)
        self.assertEqual(0, context.candidate_count)
        self.assertTrue(context.generic_fallback_eligible)

    def test_16_incomplete_preview_produces_incomplete_state(self):
        battle = _battle()
        battle.opponent.reserve = battle.opponent.reserve[:4]
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.INCOMPLETE_PREVIEW, context.match_state)

    def test_17_five_member_preview_cannot_match(self):
        battle = _battle()
        battle.opponent.reserve.pop()
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.INCOMPLETE_PREVIEW, context.match_state)
        self.assertEqual(0, context.candidate_count)

    def test_18_seven_member_preview_cannot_match(self):
        battle = _battle()
        battle.opponent.reserve.append(_pokemon("raichu"))
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.INVALID_PREVIEW, context.match_state)
        self.assertEqual(0, context.candidate_count)

    def test_19_duplicate_base_species_produces_invalid_preview(self):
        species = ("raichu", "raichualola", *DEFAULT_SPECIES[2:])
        context = _match(_battle(species=species), self.registry)
        self.assertIs(TeamPoolMatchState.INVALID_PREVIEW, context.match_state)
        self.assertIn("Species Clause", context.provenance.detail)

    def test_20_exact_forms_affect_roster_key(self):
        raichu_roster = ("raichu", *DEFAULT_SPECIES[1:])
        alola_roster = ("raichualola", *DEFAULT_SPECIES[1:])
        pool = _pool(records=(_team_record(species=raichu_roster),))
        ordinary = _match(_battle(species=raichu_roster), TeamPoolRegistry((pool,)))
        alola = _match(_battle(species=alola_roster), TeamPoolRegistry((pool,)))
        self.assertIs(TeamPoolMatchState.MATCHED, ordinary.match_state)
        self.assertIs(TeamPoolMatchState.NO_MATCH, alola.match_state)
        self.assertNotEqual(ordinary.roster_key, alola.roster_key)

    def test_21_preview_order_does_not_affect_matching(self):
        first = _match(_battle(), self.registry)
        second = _match(_battle(species=tuple(reversed(DEFAULT_SPECIES))), self.registry)
        self.assertEqual(first.roster_key, second.roster_key)
        self.assertEqual(first.candidate_ids, second.candidate_ids)

    def test_22_gender_and_cosmetic_detail_do_not_affect_matching(self):
        battle = _battle()
        for index, pokemon in enumerate(battle.opponent.reserve):
            pokemon.nickname = f"cosmetic-{index}"
            pokemon.gender = "M" if index % 2 else "F"
            pokemon.level = 50 + index
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.MATCHED, context.match_state)

    def test_23_trainer_name_does_not_affect_matching(self):
        first = _battle()
        first.opponent.account_name = "Trainer One"
        second = _battle()
        second.opponent.account_name = "Entirely Different"
        self.assertEqual(
            _match(first, self.registry).candidate_ids,
            _match(second, self.registry).candidate_ids,
        )

    def test_24_compatible_format_matches(self):
        context = _match(_battle("gen9tugs"), self.registry)
        self.assertIs(TeamPoolMatchState.MATCHED, context.match_state)
        self.assertEqual((self.pool.identity,), context.selected_pool_identities)

    def test_25_incompatible_format_pool_does_not_match(self):
        context = _match(_battle("gen9ou"), self.registry)
        self.assertIs(TeamPoolMatchState.NO_POOL, context.match_state)
        self.assertEqual((), context.selected_pool_identities)

    def test_26_ordinary_gen9_battle_is_unaffected_by_tugs_pool(self):
        battle = _battle("gen9ou")
        before = tuple(pokemon.name for pokemon in battle.opponent.reserve)
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.NO_POOL, context.match_state)
        self.assertEqual(before, tuple(pokemon.name for pokemon in battle.opponent.reserve))

    def test_27_sequential_battles_do_not_reuse_stale_candidates(self):
        mode = StandardBattleMode()
        first = _battle()
        second = _battle(species=ALTERNATE_SPECIES)
        mode.match_team_preview(first, self.registry)
        mode.match_team_preview(second, self.registry)
        self.assertEqual(1, first.team_inference.candidate_count)
        self.assertEqual(0, second.team_inference.candidate_count)
        self.assertIs(TeamPoolMatchState.NO_MATCH, second.team_inference.match_state)

    def test_28_different_registries_do_not_contaminate_each_other(self):
        first_registry = self.registry
        second_pool = _pool("secondpool", records=(_team_record("secondteam"),))
        first = _match(_battle(), first_registry)
        second = _match(_battle(), TeamPoolRegistry((second_pool,)))
        self.assertEqual("syntheticpool", first.candidate_ids[0].pool_identity.pool_id)
        self.assertEqual("secondpool", second.candidate_ids[0].pool_identity.pool_id)

    def test_29_context_creation_and_matching_do_not_mutate_pool_records(self):
        teams_before = self.pool.teams
        index_before = tuple(self.pool.roster_index.items())
        metadata_before = repr(self.pool.metadata)
        _match(_battle(), self.registry)
        self.assertIs(teams_before, self.pool.teams)
        self.assertEqual(index_before, tuple(self.pool.roster_index.items()))
        self.assertEqual(metadata_before, repr(self.pool.metadata))

    def test_30_matching_does_not_mutate_global_pokedex_or_move_data(self):
        moves_before = copy.deepcopy(all_move_json)
        pokedex_before = copy.deepcopy(pokedex)
        _match(_battle(), self.registry)
        self.assertEqual(moves_before, all_move_json)
        self.assertEqual(pokedex_before, pokedex)

    def test_31_matching_does_not_mutate_team_datasets(self):
        battle = _battle()
        mode = battle.mode
        before = dict(vars(mode.team_datasets))
        mode.match_team_preview(battle, self.registry)
        self.assertEqual(before, vars(mode.team_datasets))

    def test_32_matching_does_not_mutate_smogon_sets(self):
        battle = _battle()
        mode = battle.mode
        before = dict(vars(mode.smogon_sets))
        mode.match_team_preview(battle, self.registry)
        self.assertEqual(before, vars(mode.smogon_sets))

    def test_33_own_team_state_remains_unchanged(self):
        battle = _battle()
        own_active = battle.user.active
        own_reserve = tuple(battle.user.reserve)
        own_snapshot = (
            own_active.name,
            own_active.item,
            own_active.ability,
            tuple(own_active.moves),
            own_active.nature,
            tuple(own_active.evs),
        )
        _match(battle, self.registry)
        self.assertIs(own_active, battle.user.active)
        self.assertEqual(own_reserve, tuple(battle.user.reserve))
        self.assertEqual(
            own_snapshot,
            (
                own_active.name,
                own_active.item,
                own_active.ability,
                tuple(own_active.moves),
                own_active.nature,
                tuple(own_active.evs),
            ),
        )

    def test_34_opponent_state_gains_no_unrevealed_item(self):
        battle = _battle()
        _match(battle, self.registry)
        self.assertTrue(
            all(pokemon.item == constants.UNKNOWN_ITEM for pokemon in battle.opponent.reserve)
        )

    def test_35_opponent_state_gains_no_unrevealed_ability(self):
        battle = _battle()
        _match(battle, self.registry)
        self.assertTrue(all(pokemon.ability is None for pokemon in battle.opponent.reserve))

    def test_36_opponent_state_gains_no_unrevealed_moves(self):
        battle = _battle()
        _match(battle, self.registry)
        self.assertTrue(all(not pokemon.moves for pokemon in battle.opponent.reserve))

    def test_37_opponent_state_gains_no_pool_nature_or_evs(self):
        battle = _battle()
        pool_record = self.pool.teams[0].pokemon[0]
        _match(battle, self.registry)
        self.assertTrue(all(pokemon.nature != pool_record.nature_id for pokemon in battle.opponent.reserve))
        self.assertTrue(all(tuple(pokemon.evs) != pool_record.evs.as_tuple() for pokemon in battle.opponent.reserve))

    def test_38_no_poke_engine_serialization_schema_change_exists(self):
        battle = _battle()
        _match(battle, self.registry)
        battle.opponent.active = battle.opponent.reserve.pop(0)
        state = battle_to_poke_engine_state(battle)
        self.assertFalse(hasattr(state, "team_inference"))
        self.assertNotIn("team_inference", state.to_string())

    def test_39_open_policy_is_placeholder_without_exact_population(self):
        battle = _battle(policy=TeamSheetPolicy.OPEN)
        before = [(pokemon.item, pokemon.ability, tuple(pokemon.moves)) for pokemon in battle.opponent.reserve]
        context = _match(battle, self.registry)
        after = [(pokemon.item, pokemon.ability, tuple(pokemon.moves)) for pokemon in battle.opponent.reserve]
        self.assertIs(TeamSheetPolicy.OPEN, context.policy)
        self.assertFalse(context.exact_fields_available)
        self.assertEqual(before, after)

    def test_40_existing_preview_initialization_works_without_registry(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.user.active = _pokemon("weedle")
        battle.initialize_team_preview(
            [f"{species_id}, L100, {'M' if index % 2 else 'F'}" for index, species_id in enumerate(DEFAULT_SPECIES)],
            "gen9tugs",
        )
        context = _match(battle, None)
        self.assertEqual(6, len(battle.opponent.reserve))
        self.assertIs(TeamPoolMatchState.NO_POOL, context.match_state)

    def test_unresolved_prefix_form_is_invalid_instead_of_matching(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.user.active = _pokemon("weedle")
        preview = ["Pikachu-Unresolved, L100"] + [
            f"{species_id}, L100" for species_id in DEFAULT_SPECIES[1:]
        ]
        battle.initialize_team_preview(preview, "gen9tugs")
        context = _match(battle, self.registry)
        self.assertTrue(battle.opponent.reserve[0].unknown_forme)
        self.assertIs(TeamPoolMatchState.INVALID_PREVIEW, context.match_state)

    def test_unknown_exact_form_is_invalid_without_preview_crash(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.user.active = _pokemon("weedle")
        preview = ["EntirelyUnknownMon, L100"] + [
            f"{species_id}, L100" for species_id in DEFAULT_SPECIES[1:]
        ]
        battle.initialize_team_preview(preview, "gen9tugs")
        context = _match(battle, self.registry)
        self.assertIs(TeamPoolMatchState.INVALID_PREVIEW, context.match_state)

    def test_safe_summary_and_observation_placeholder_are_immutable(self):
        context = _match(_battle(), self.registry)
        summary = context.safe_summary()
        self.assertIsInstance(summary.observation_filter_state, ObservationFilterState)
        with self.assertRaises(AttributeError):
            summary.observation_filter_state.revision = 1


class TestPreviewLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_41_existing_preview_lifecycle_works_with_registry(self):
        mode = _PreviewLifecycleMode()
        registry = TeamPoolRegistry((_pool(),))
        with patch(
            "fp.modes.standard_battle.get_first_request_json",
            new=_initialize_synthetic_user,
        ):
            battle = await mode.start_battle(
                None,
                "gen9tugs",
                {},
                team_pool_registry=registry,
            )
        self.assertTrue(mode.datasets_initialized)
        self.assertTrue(mode.preview_handled)
        self.assertEqual(6, len(battle.opponent.reserve))
        self.assertIs(TeamPoolMatchState.MATCHED, battle.team_inference.match_state)


class TestExistingTugsRegressions(unittest.TestCase):
    def test_42_existing_crag_mend_behavior_is_unaffected(self):
        moves_before = copy.deepcopy(all_move_json)
        pokedex_before = copy.deepcopy(pokedex)
        try:
            apply_mods(FormatSpec.from_format_string("gen9tugs"))
            self.assertTrue(unlikely_to_have_choice_item("cragmend"))
            self.assertEqual(5, all_move_json["cragmend"][constants.PP])
        finally:
            all_move_json.clear()
            all_move_json.update(moves_before)
            pokedex.clear()
            pokedex.update(pokedex_before)

    def test_43_existing_trick_room_persistent_behavior_is_unaffected(self):
        battle = Battle(None)
        battle.pokemon_format = "gen9tugs"
        battle.generation = "gen9"
        battle.turn = 3
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("lapras")
        battle.user.active.ability = "persistent"
        battle.opponent.active = _pokemon("pikachu")
        fieldstart(
            battle,
            ["", "-fieldstart", "move: Trick Room", "[of] p1a: Lapras", "[persistent]"],
        )
        self.assertTrue(battle.trick_room)
        self.assertEqual(8, battle.trick_room_turns_remaining)

    def test_44_existing_ancient_shell_serialization_is_unaffected(self):
        pokemon = _pokemon("lapras")
        pokemon.ability = "ancientshell"
        pokemon.item = "leftovers"
        serialized = pokemon_to_poke_engine_pkmn(pokemon)
        self.assertEqual("ancientshell", serialized.ability)

    def test_45_existing_corrosive_gas_parsing_is_unaffected(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("pikachu")
        battle.opponent.active = _pokemon("dustox")
        battle.opponent.active.item = "leftovers"
        remove_item(
            battle,
            ["", "-enditem", "p2a: Dustox", "Leftovers", "[from] move: Corrosive Gas"],
        )
        self.assertIsNone(battle.opponent.active.item)
        self.assertEqual("leftovers", battle.opponent.active.removed_item)

    def test_46_existing_closing_jaws_behavior_is_unaffected(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = _pokemon("pikachu")
        battle.opponent.active = _pokemon("mawile")
        activate(
            battle,
            ["", "-activate", "p2a: Mawile", "ability: Closing Jaws"],
        )
        self.assertEqual("closingjaws", battle.opponent.active.ability)


if __name__ == "__main__":
    unittest.main()
