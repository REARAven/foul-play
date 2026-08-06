import asyncio
import copy
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fp import constants
from fp.battle.public_prior_context import PublicPriorFallback
from fp.battle.protocol import process_battle_updates
from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig, SaveReplay
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import PublicPriorIdentity, load_public_prior
from fp.data.public_priors.runtime import (
    PublicPriorStartupOptions,
    load_public_prior_runtime_configuration,
)
from fp.format_spec import FormatSpec
from fp.modes import battle_mode
from fp.modes.standard_battle import StandardBattleMode
from fp.run_battle import pokemon_battle, start_battle
from fp.search.standard_battles import prepare_battles
from fp.websocket_client import PSWebsocketClient


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATH = (
    ROOT
    / "fp"
    / "data"
    / "public_priors"
    / "pools"
    / "tugspublicarchetypes-1.0.0.json"
)
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.0.0", "gen9tugs")
OPPONENT_PREVIEW = (
    "aerodactyl",
    "porygon2",
    "mawile",
    "forretress",
    "lapras",
    "druddigon",
)
USER_TEAM = (
    "porygon2",
    "sylveon",
    "vikavolt",
    "jellicent",
    "obstagoon",
    "forretress",
)

ORIGINAL_MOVES = copy.deepcopy(all_move_json)
ORIGINAL_POKEDEX = copy.deepcopy(pokedex)
ORIGINAL_FORMAT = FoulPlayConfig.pokemon_format
ORIGINAL_USERNAME = getattr(FoulPlayConfig, "username", None)
ORIGINAL_LOG_TO_FILE = getattr(FoulPlayConfig, "log_to_file", None)
ORIGINAL_BATTLE_TIMER = FoulPlayConfig.battle_timer
ORIGINAL_SAVE_REPLAY = getattr(FoulPlayConfig, "save_replay", None)


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"
    FoulPlayConfig.username = "LifecycleBot"
    FoulPlayConfig.log_to_file = False
    FoulPlayConfig.battle_timer = True
    FoulPlayConfig.save_replay = SaveReplay.never
    apply_mods(FormatSpec.from_format_string("gen9tugs"))


def tearDownModule():
    all_move_json.clear()
    all_move_json.update(ORIGINAL_MOVES)
    pokedex.clear()
    pokedex.update(ORIGINAL_POKEDEX)
    FoulPlayConfig.pokemon_format = ORIGINAL_FORMAT
    FoulPlayConfig.username = ORIGINAL_USERNAME
    FoulPlayConfig.log_to_file = ORIGINAL_LOG_TO_FILE
    FoulPlayConfig.battle_timer = ORIGINAL_BATTLE_TIMER
    FoulPlayConfig.save_replay = ORIGINAL_SAVE_REPLAY


def _runtime(fallback):
    return load_public_prior_runtime_configuration(
        PublicPriorStartupOptions((str(PRODUCTION_PATH),), fallback),
        "gen9tugs",
    )


def _preview_message():
    lines = [">battle-gen9tugs-209", "|clearpoke"]
    lines.extend("|poke|p1|{}, L100".format(species) for species in OPPONENT_PREVIEW)
    lines.extend("|poke|p2|{}, L100".format(species) for species in USER_TEAM)
    lines.append("|teampreview")
    return "\n".join(lines)


async def _initialize_user(_websocket, battle):
    battle.user.active = Pokemon(USER_TEAM[0], 100)
    battle.user.reserve = [Pokemon(species, 100) for species in USER_TEAM[1:]]
    battle.rqid = 3
    battle.request_json = {constants.RQID: 3, "teamPreview": True}


class _FakeWebsocket:
    accept_challenge = PSWebsocketClient.accept_challenge

    def __init__(self):
        self.username = "LifecycleBot"
        self.events = []
        self.messages = [
            "|pm|Opponent|LifecycleBot|/challenge|gen9tugs|||",
            ">battle-gen9tugs-209|init|battle|title|Opponent vs. LifecycleBot",
            ">battle-gen9tugs-209\n|player|p1|Opponent|1|",
            _preview_message(),
        ]
        self.sent = []
        self.joined = []

    async def login(self):
        self.events.append("login")
        return self.username

    async def receive_message(self):
        if not self.messages:
            raise AssertionError("synthetic websocket message queue exhausted")
        return self.messages.pop(0)

    async def send_message(self, room, messages):
        self.sent.append((room, tuple(messages)))
        if messages and messages[0].startswith("/accept "):
            self.events.append("accept")

    async def join_room(self, room_name):
        self.joined.append(room_name)


class _LifecycleMode(StandardBattleMode):
    def __init__(self):
        super().__init__()
        self.team_datasets.initialize = mock.Mock()
        self.smogon_sets.initialize = mock.Mock()
        self.preview_context = None
        self.preview_ledger = None

    async def handle_team_preview(self, battle, websocket):
        self.preview_context = battle.public_prior_context
        self.preview_ledger = battle.team_inference.observation_ledger


async def _accept_and_start(configuration):
    websocket = _FakeWebsocket()
    mode = _LifecycleMode()
    await websocket.login()
    await websocket.accept_challenge("gen9tugs", None)
    with mock.patch("fp.run_battle.battle_mode", return_value=mode), mock.patch(
        "fp.modes.standard_battle.get_first_request_json",
        new=_initialize_user,
    ):
        battle = await start_battle(
            websocket,
            "gen9tugs",
            {},
            public_prior_configuration=configuration,
        )
    return battle, mode, websocket


def _activate_druddigon_and_record_live_evidence(battle):
    battle.msg_list = [
        "|switch|p1a: Druddigon|Druddigon, M|100/100",
        "|switch|p2a: Porygon2|Porygon2|100/100",
        "|-ability|p2a: Porygon2|Rough Skin|Trace|[from] ability: Trace|[of] p1a: Druddigon",
        "|move|p1a: Druddigon|Glare|p2a: Porygon2",
    ]
    process_battle_updates(battle)


def _find_opponent(battle, species_id):
    if battle.opponent.active is not None and battle.opponent.active.name == species_id:
        return battle.opponent.active
    return battle.opponent.find_pokemon_in_reserves(species_id)


def _automatic_timer_messages(websocket):
    return [
        (room, messages)
        for room, messages in websocket.sent
        if messages in (("/timer on",), ("/timer off",))
    ]


class TestBattleTimerRuntime(unittest.TestCase):
    def tearDown(self):
        FoulPlayConfig.battle_timer = True

    def test_08_default_runtime_sends_exactly_one_timer_on(self):
        FoulPlayConfig.battle_timer = True
        _, _, websocket = asyncio.run(_accept_and_start(None))
        self.assertEqual(
            [("battle-gen9tugs-209", ("/timer on",))],
            _automatic_timer_messages(websocket),
        )

    def test_09_explicit_on_runtime_sends_exactly_one_timer_on(self):
        FoulPlayConfig.battle_timer = True
        _, _, websocket = asyncio.run(
            _accept_and_start(_runtime(PublicPriorFallback.NONE))
        )
        self.assertEqual(
            [("battle-gen9tugs-209", ("/timer on",))],
            _automatic_timer_messages(websocket),
        )

    def test_10_off_runtime_sends_no_timer_on(self):
        FoulPlayConfig.battle_timer = False
        _, _, websocket = asyncio.run(_accept_and_start(None))
        self.assertNotIn(
            ("battle-gen9tugs-209", ("/timer on",)), websocket.sent
        )

    def test_11_off_runtime_sends_no_timer_off(self):
        FoulPlayConfig.battle_timer = False
        _, _, websocket = asyncio.run(_accept_and_start(None))
        self.assertNotIn(
            ("battle-gen9tugs-209", ("/timer off",)), websocket.sent
        )

    def test_12_repeated_battle_updates_do_not_duplicate_timer_on(self):
        FoulPlayConfig.battle_timer = True
        battle, _, websocket = asyncio.run(_accept_and_start(None))
        for seconds in (135, 60, 30):
            battle.msg_list = [
                "|inactive|Time left: {} sec this turn|{} sec total".format(
                    seconds, seconds
                )
            ]
            process_battle_updates(battle)
        self.assertEqual(30, battle.time_remaining)
        self.assertEqual(1, len(_automatic_timer_messages(websocket)))

    def test_15_incoming_timer_protocol_still_processes_when_off(self):
        FoulPlayConfig.battle_timer = False
        battle, _, websocket = asyncio.run(_accept_and_start(None))
        battle.msg_list = [
            "|inactive|Time left: 60 sec this turn|60 sec total"
        ]
        process_battle_updates(battle)
        self.assertEqual(60, battle.time_remaining)
        battle.msg_list = ["|inactiveoff|Battle timer is now OFF."]
        process_battle_updates(battle)
        self.assertIsNone(battle.time_remaining)
        self.assertEqual([], _automatic_timer_messages(websocket))

    def test_16_opponent_started_timer_does_not_trigger_counter_command(self):
        FoulPlayConfig.battle_timer = False
        battle, _, websocket = asyncio.run(_accept_and_start(None))
        before = tuple(websocket.sent)
        battle.msg_list = [
            "|inactive|Battle timer is ON: inactive players will automatically lose when time's up."
        ]
        process_battle_updates(battle)
        self.assertEqual(before, tuple(websocket.sent))
        self.assertEqual([], _automatic_timer_messages(websocket))

    def test_23_battle_cleanup_emits_no_additional_timer_command(self):
        class FinishedMode:
            async def start_battle(self, websocket, battle_format, team_dict, **kwargs):
                return SimpleNamespace(battle_tag="battle-finished")

        class FinishedSocket:
            def __init__(self):
                self.sent = []
                self.left = []

            async def send_message(self, room, messages):
                self.sent.append((room, tuple(messages)))

            async def receive_message(self):
                return ">battle-finished\n|win|LifecycleBot\n"

            async def leave_battle(self, battle_tag):
                self.left.append(battle_tag)

        FoulPlayConfig.battle_timer = True
        websocket = FinishedSocket()
        with mock.patch("fp.run_battle.battle_mode", return_value=FinishedMode()):
            winner = asyncio.run(
                pokemon_battle(websocket, "gen9tugs", None)
            )
        self.assertEqual("LifecycleBot", winner)
        self.assertEqual(["battle-finished"], websocket.left)
        self.assertEqual(
            [("battle-finished", ("/timer on",))],
            _automatic_timer_messages(websocket),
        )

    def test_24_repeated_initialization_of_one_room_is_idempotent(self):
        class SameRoomMode:
            async def start_battle(self, websocket, battle_format, team_dict, **kwargs):
                return SimpleNamespace(battle_tag="battle-repeated")

        FoulPlayConfig.battle_timer = True
        websocket = _FakeWebsocket()
        with mock.patch("fp.run_battle.battle_mode", return_value=SameRoomMode()):
            asyncio.run(start_battle(websocket, "gen9tugs", None))
            asyncio.run(start_battle(websocket, "gen9tugs", None))
        self.assertEqual(
            [("battle-repeated", ("/timer on",))],
            _automatic_timer_messages(websocket),
        )


class TestLivePublicPriorLifecycle(unittest.TestCase):
    def test_01_real_production_runtime_loads_once_and_creates_fresh_contexts(self):
        import fp.data.public_priors.runtime as runtime_module

        with mock.patch.object(
            runtime_module,
            "load_public_prior",
            wraps=load_public_prior,
        ) as loader:
            configuration = _runtime(PublicPriorFallback.GENERIC)
        loader.assert_called_once_with(PRODUCTION_PATH)
        first = configuration.create_battle_context("gen9tugs")
        second = configuration.create_battle_context("gen9tugs")
        self.assertIsNot(first, second)
        self.assertIs(first.registry, second.registry)
        self.assertEqual((IDENTITY,), first.selected_identities)
        self.assertIs(PublicPriorFallback.GENERIC, first.fallback_policy)

    def test_02_accept_challenge_reaches_real_battle_start_and_repairs_battle_209(self):
        configuration = _runtime(PublicPriorFallback.GENERIC)
        with mock.patch("fp.modes.base.logger.info") as lifecycle_logger:
            battle, mode, websocket = asyncio.run(_accept_and_start(configuration))

        context = battle.public_prior_context
        self.assertIsNotNone(context)
        self.assertIs(context, mode.preview_context)
        self.assertIs(
            mode.preview_ledger,
            battle.team_inference.observation_ledger,
        )
        self.assertEqual("gen9tugs", context.format_id)
        self.assertEqual((IDENTITY,), context.selected_identities)
        self.assertIs(PublicPriorFallback.GENERIC, context.fallback_policy)
        self.assertEqual("gen9tugs", battle.pokemon_format)
        self.assertEqual("gen9tugs", context.registry.get(IDENTITY).identity.format_id)
        self.assertEqual(OPPONENT_PREVIEW, tuple(member.species_id for member in context_ledger_sorted(battle)))
        self.assertIn(("", ("/accept Opponent",)), websocket.sent)
        self.assertEqual(["login", "accept"], websocket.events)

        attached = [
            call.args[0]
            for call in lifecycle_logger.call_args_list
            if "Public prior context attached:" in call.args[0]
        ]
        self.assertEqual(1, len(attached))
        self.assertIn("format=gen9tugs datasets=1 fallback=generic", attached[0])

        _activate_druddigon_and_record_live_evidence(battle)
        evidence = battle.team_inference.observation_ledger.member("druddigon")
        self.assertEqual("roughskin", evidence.base_ability_id)
        self.assertEqual(("glare",), evidence.selected_move_ids)

        canonical = _pokemon_state(battle.opponent.active)
        generic_species = []

        def generic(pokemon, _mode):
            generic_species.append(pokemon.name)

        with mock.patch("fp.search.standard_battles.logger.debug") as sampling_logger, mock.patch(
            "fp.search.standard_battles.sample_pokemon", side_effect=generic
        ):
            sampled = prepare_battles(battle, 1)[0][0]

        sampled_druddigon = _find_opponent(sampled, "druddigon")
        self.assertEqual("rockyhelmet", sampled_druddigon.item)
        self.assertEqual(("roughskin", "roughskin"), (sampled_druddigon.ability, sampled_druddigon.original_ability))
        self.assertEqual("glare", sampled_druddigon.moves[0].name)
        self.assertEqual(
            {"cragmend", "glare", "dragontail", "stealthrock"},
            {move.name for move in sampled_druddigon.moves},
        )
        self.assertEqual(canonical, _pokemon_state(battle.opponent.active))
        self.assertEqual(["aerodactyl"], generic_species)

        diagnostics = "\n".join(
            call.args[0] for call in sampling_logger.call_args_list
        )
        self.assertIn(
            "Public prior selected: species=druddigon dataset=tugspublicarchetypes version=1.0.0 variant=roughskinutility",
            diagnostics,
        )
        self.assertIn("Public prior miss: species=aerodactyl fallback=generic", diagnostics)
        for private_value in (
            "rockyhelmet",
            "cragmend",
            "impish",
            "252",
        ):
            self.assertNotIn(private_value, diagnostics)

    def test_03_context_and_observations_survive_copy_with_correct_ownership(self):
        configuration = _runtime(PublicPriorFallback.GENERIC)
        battle, _, _ = asyncio.run(_accept_and_start(configuration))
        copied = copy.deepcopy(battle)
        self.assertIs(battle.public_prior_context, copied.public_prior_context)
        self.assertIs(
            battle.public_prior_context.registry,
            copied.public_prior_context.registry,
        )
        self.assertIsNot(battle.team_inference, copied.team_inference)
        self.assertIsNot(
            battle.team_inference.observation_ledger,
            copied.team_inference.observation_ledger,
        )
        copied.team_inference.record_selected_move("druddigon", "glare")
        self.assertEqual((), battle.team_inference.observation_ledger.member("druddigon").selected_move_ids)
        self.assertEqual(("glare",), copied.team_inference.observation_ledger.member("druddigon").selected_move_ids)
        self.assertFalse(
            {"__copy__", "__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"}
            & set(Battle.__dict__)
        )

    def test_04_prepared_battles_have_independent_mutable_pokemon(self):
        battle, _, _ = asyncio.run(
            _accept_and_start(_runtime(PublicPriorFallback.GENERIC))
        )
        _activate_druddigon_and_record_live_evidence(battle)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepared = prepare_battles(battle, 2)
        self.assertEqual(
            ["aerodactyl", "aerodactyl"],
            [call.args[0].name for call in generic.call_args_list],
        )
        first, second = [entry[0] for entry in prepared]
        first_druddigon = _find_opponent(first, "druddigon")
        second_druddigon = _find_opponent(second, "druddigon")
        self.assertIsNot(first_druddigon, second_druddigon)
        first_druddigon.hp = 1
        self.assertNotEqual(first_druddigon.hp, second_druddigon.hp)
        self.assertEqual([0.5, 0.5], [entry[1] for entry in prepared])

    def test_05_sequential_battles_have_fresh_contexts_and_ledgers(self):
        configuration = _runtime(PublicPriorFallback.GENERIC)
        first, _, _ = asyncio.run(_accept_and_start(configuration))
        second, _, _ = asyncio.run(_accept_and_start(configuration))
        self.assertIsNot(first.public_prior_context, second.public_prior_context)
        self.assertIs(
            first.public_prior_context.registry,
            second.public_prior_context.registry,
        )
        self.assertIsNot(first.team_inference, second.team_inference)
        first.team_inference.record_selected_move("druddigon", "glare")
        self.assertEqual((), second.team_inference.observation_ledger.member("druddigon").selected_move_ids)

    def test_06_none_lifecycle_is_public_only_and_cache_network_free(self):
        configuration = _runtime(PublicPriorFallback.NONE)
        with mock.patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("network attempted"),
        ):
            battle, mode, _ = asyncio.run(_accept_and_start(configuration))
            _activate_druddigon_and_record_live_evidence(battle)
            with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
                sampled = prepare_battles(battle, 1)[0][0]

        self.assertIs(PublicPriorFallback.NONE, battle.public_prior_context.fallback_policy)
        mode.team_datasets.initialize.assert_not_called()
        mode.smogon_sets.initialize.assert_not_called()
        generic.assert_not_called()
        self.assertEqual("rockyhelmet", _find_opponent(sampled, "druddigon").item)
        self.assertEqual(constants.UNKNOWN_ITEM, _find_opponent(sampled, "aerodactyl").item)

    def test_07_no_configuration_preserves_legacy_generic_behavior(self):
        battle, mode, _ = asyncio.run(_accept_and_start(None))
        self.assertIsNone(battle.public_prior_context)
        mode.team_datasets.initialize.assert_called_once()
        mode.smogon_sets.initialize.assert_called_once()
        battle.opponent.active = battle.opponent.reserve.pop(0)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(battle, 1)
        self.assertEqual(6, generic.call_count)

    def test_08_live_format_representations_stay_canonical(self):
        configuration = _runtime(PublicPriorFallback.GENERIC)
        battle, _, _ = asyncio.run(_accept_and_start(configuration))
        dataset = configuration.registry.get(IDENTITY)
        self.assertEqual(
            ("gen9tugs", "gen9tugs", "gen9tugs", "gen9tugs"),
            (
                FoulPlayConfig.pokemon_format,
                configuration.format_id,
                battle.pokemon_format,
                dataset.identity.format_id,
            ),
        )
        self.assertNotEqual("[Gen 9] TUGS", battle.pokemon_format)

    def test_09_public_search_has_no_private_or_global_runtime_dependency(self):
        paths = (
            ROOT / "fp" / "search" / "standard_battles.py",
            ROOT / "fp" / "search" / "public_prior_sampling.py",
            ROOT / "fp" / "battle" / "public_prior_context.py",
            ROOT / "fp" / "data" / "public_priors" / "runtime.py",
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        forbidden = (
            "fp.data.team_pools",
            "TeamPool",
            "TeamRecord",
            "PokemonRecord",
            "TeamPoolCandidateId",
            "baseline_candidate_ids",
            "active_candidate_ids",
        )
        self.assertEqual(set(), {token for token in forbidden if token in source})
        mode = battle_mode(FormatSpec.from_format_string("gen9tugs").battle_type)
        self.assertFalse(hasattr(mode, "public_prior_context"))
        self.assertFalse(hasattr(mode, "public_prior_configuration"))


def context_ledger_sorted(battle):
    return tuple(
        sorted(
            battle.team_inference.observation_ledger.members,
            key=lambda member: OPPONENT_PREVIEW.index(member.species_id),
        )
    )


def _pokemon_state(pokemon):
    return (
        pokemon.item,
        pokemon.ability,
        pokemon.original_ability,
        tuple(pokemon.moves),
        pokemon.nature,
        tuple(pokemon.evs),
        tuple(pokemon.ivs),
    )


if __name__ == "__main__":
    unittest.main()
