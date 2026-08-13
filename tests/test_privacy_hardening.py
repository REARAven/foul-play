import asyncio
from contextlib import ExitStack
import io
import json
import logging
from pathlib import Path
import traceback
from types import SimpleNamespace
import unittest
from unittest import mock

from fp import constants
from fp.battle.protocol import inactive, request, switch
from fp.battle.state import Battle, Battler, LastUsedMove, Pokemon
from fp.config import BotModes, CustomFormatter, FoulPlayConfig
import fp.main as main_module
from fp.modes.base import get_first_request_json
from fp.modes.standard_battle import StandardBattleMode
import fp.search.bss as search_bss
import fp.search.helpers as search_helpers
import fp.search.main as search_main
import fp.search.poke_engine_helpers as poke_engine_helpers
from fp.websocket_client import PSWebsocketClient


PRIVATE_SENTINEL = "TUGS_PRIVATE_TEAM_SENTINEL_7F4E91C2"
ROOT = Path(__file__).resolve().parents[1]


class _FakeWebsocket:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []

    async def recv(self):
        return self.messages.pop(0)

    async def send(self, message):
        self.sent.append(message)


def _client(messages=()):
    client = PSWebsocketClient()
    client.websocket = _FakeWebsocket(messages)
    client.last_message = None
    return client


def _formatted_traceback(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


class TestWebsocketPrivacy(unittest.TestCase):
    def test_team_upload_is_transmitted_without_logging_or_retention(self):
        client = _client()

        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            asyncio.run(client.update_team(PRIVATE_SENTINEL))

        output = "\n".join(captured.output)
        self.assertEqual([f"|/utm {PRIVATE_SENTINEL}"], client.websocket.sent)
        self.assertIsNone(client.last_message)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("Showdown team submitted", output)

    def test_team_upload_is_safe_with_the_application_log_formatter(self):
        client = _client()
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(CustomFormatter())
        websocket_logger = logging.getLogger("fp.websocket_client")
        previous_level = websocket_logger.level
        websocket_logger.setLevel(logging.DEBUG)
        websocket_logger.addHandler(handler)
        try:
            asyncio.run(client.update_team(PRIVATE_SENTINEL))
        finally:
            websocket_logger.removeHandler(handler)
            websocket_logger.setLevel(previous_level)

        self.assertNotIn(PRIVATE_SENTINEL, output.getvalue())
        self.assertIn("Showdown team submitted", output.getvalue())

    def test_non_sensitive_outbound_message_remains_logged_and_retained(self):
        client = _client()

        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            asyncio.run(
                client.send_message("battle-gen9tugs-safe", ["/move 1|7"])
            )

        expected = "battle-gen9tugs-safe|/move 1|7"
        self.assertEqual([expected], client.websocket.sent)
        self.assertEqual(expected, client.last_message)
        self.assertIn(expected, "\n".join(captured.output))

    def test_request_events_are_redacted_without_mutating_received_message(self):
        cases = (
            (
                f'|request|{{"private":"{PRIVATE_SENTINEL}"}}',
                "|request|<redacted>",
                1,
            ),
            (
                f'>battle-gen9tugs-safe|request|{{"private":"{PRIVATE_SENTINEL}"}}',
                ">battle-gen9tugs-safe|request|<redacted>",
                1,
            ),
            (
                ">battle-gen9tugs-safe\n"
                "|turn|4\n"
                f'|request|{{"private":"{PRIVATE_SENTINEL}"}}\n'
                "|c|SafeUser|visible diagnostic\n"
                f"|request|malformed-{PRIVATE_SENTINEL}",
                "|c|SafeUser|visible diagnostic",
                2,
            ),
        )
        for message, expected_log_fragment, redaction_count in cases:
            with self.subTest(message_kind=expected_log_fragment):
                client = _client((message,))
                with self.assertLogs(
                    "fp.websocket_client", logging.DEBUG
                ) as captured:
                    received = asyncio.run(client.receive_message())

                output = "\n".join(captured.output)
                self.assertEqual(message, received)
                self.assertNotIn(PRIVATE_SENTINEL, output)
                self.assertIn(expected_log_fragment, output)
                self.assertEqual(
                    redaction_count, output.count("|request|<redacted>")
                )
                if "|turn|4" in message:
                    self.assertIn("|turn|4", output)


class TestRequestPrivacy(unittest.TestCase):
    def test_parsed_request_logging_contains_only_safe_structural_metadata(self):
        request_document = {
            constants.RQID: 17,
            constants.WAIT: False,
            constants.FORCE_SWITCH: [False],
            constants.ACTIVE: [{"private": PRIVATE_SENTINEL}],
            constants.SIDE: {
                constants.POKEMON: [
                    {"private": PRIVATE_SENTINEL},
                    {"private": PRIVATE_SENTINEL},
                ]
            },
        }
        battle = SimpleNamespace(turn=4)

        with self.assertLogs("fp.battle.protocol", logging.DEBUG) as captured:
            request(battle, ["", "request", json.dumps(request_document)])

        output = "\n".join(captured.output)
        self.assertEqual(request_document, battle.request_json)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("rqid=17", output)
        self.assertIn("pokemon_count=2", output)
        self.assertIn("active_slots=1", output)
        self.assertIn("turn=4", output)

    def test_non_numeric_request_id_cannot_enter_logs(self):
        request_document = {
            constants.RQID: PRIVATE_SENTINEL,
            constants.SIDE: {constants.POKEMON: []},
        }
        battle = SimpleNamespace(turn=0)

        with self.assertLogs("fp.battle.protocol", logging.DEBUG) as captured:
            request(battle, ["", "request", json.dumps(request_document)])

        output = "\n".join(captured.output)
        self.assertEqual(PRIVATE_SENTINEL, battle.rqid)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("rqid=unknown", output)

    def test_malformed_timer_protocol_does_not_log_raw_payload(self):
        battle = SimpleNamespace(time_remaining=None)
        with self.assertLogs("fp.battle.protocol", logging.WARNING) as captured:
            inactive(
                battle,
                ["", "inactive", f"Time left: {PRIVATE_SENTINEL}"],
            )

        output = "\n".join(captured.output)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("did not match expected format", output)

    def test_malformed_request_exception_and_traceback_are_sanitized(self):
        malformed = f"malformed-{PRIVATE_SENTINEL}"

        with self.assertRaises(ValueError) as captured:
            request(SimpleNamespace(turn=0), ["", "request", malformed])

        self.assertEqual(
            "Could not parse Showdown request payload", str(captured.exception)
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_request_reinitialization_exception_does_not_embed_request(self):
        battler = Battler()
        with mock.patch.object(
            FoulPlayConfig, "pokemon_format", "gen9tugs", create=True
        ):
            battler.active = Pokemon("pikachu", 100)
        private_entry = {
            constants.DETAILS: "Pikachu, L100",
            "private": PRIVATE_SENTINEL,
        }
        request_document = {
            constants.SIDE: {
                constants.POKEMON: [private_entry, private_entry.copy()]
            }
        }

        with self.assertRaises(ValueError) as captured:
            battler.re_initialize_active_pokemon_from_request_json(request_document)

        self.assertEqual(
            "Could not reinitialize active Pokemon from Showdown request",
            str(captured.exception),
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_initial_request_parser_exception_and_traceback_are_sanitized(self):
        class Client:
            async def receive_message(self):
                return f"|request|malformed-{PRIVATE_SENTINEL}"

        battle = SimpleNamespace()
        with self.assertRaises(ValueError) as captured:
            asyncio.run(get_first_request_json(Client(), battle))

        self.assertEqual(
            "Could not parse Showdown request payload", str(captured.exception)
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_private_illusion_request_details_are_not_logged(self):
        with mock.patch.object(
            FoulPlayConfig, "pokemon_format", "gen9tugs", create=True
        ):
            battle = Battle("battle-gen9tugs-safe")
            battle.generation = "gen9"
            battle.mode = StandardBattleMode()
            battle.user.name = "p1"
            battle.opponent.name = "p2"
            battle.user.active = Pokemon("pikachu", 100)
            battle.user.reserve = [Pokemon("zoroark", 100), Pokemon("weedle", 100)]
            battle.user.last_selected_move = LastUsedMove(
                "pikachu", "switch zoroark", 0
            )
            battle.request_json = {
                constants.SIDE: {
                    constants.POKEMON: [
                        {
                            constants.IDENT: f"p1: {PRIVATE_SENTINEL}",
                            constants.DETAILS: "Zoroark, L100, M",
                            constants.ACTIVE: True,
                        }
                    ]
                }
            }

            with self.assertLogs("fp.battle.protocol", logging.INFO) as captured:
                switch(
                    battle,
                    ["", "switch", "p1a: Weedle", "Weedle, L100, M", "100/100"],
                )

        output = "\n".join(captured.output)
        self.assertEqual("zoroark", battle.user.active.name)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("Resolved a user illusion switch", output)

    def test_unknown_selected_team_details_are_not_logged(self):
        with mock.patch.object(
            FoulPlayConfig, "pokemon_format", "gen9tugs", create=True
        ), self.assertLogs("fp.battle.state", logging.DEBUG) as captured:
            pokemon = Pokemon(f"Pikachu-{PRIVATE_SENTINEL}", 100)
            pokemon.add_move(PRIVATE_SENTINEL)

        output = "\n".join(captured.output)
        self.assertEqual("pikachu", pokemon.name)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("unrecognized Pokemon form", output)
        self.assertIn("unknown move", output)

    def test_request_state_conversion_exception_is_sanitized(self):
        request_document = {
            constants.SIDE: {
                constants.ID: "p1",
                constants.POKEMON: [
                    {
                        constants.IDENT: "p1: Pikachu",
                        constants.DETAILS: "Pikachu, L100",
                        constants.CONDITION: PRIVATE_SENTINEL,
                        constants.ACTIVE: True,
                        constants.STATS: {},
                        constants.MOVES: [],
                        constants.ITEM: "",
                        constants.ABILITY: "static",
                    }
                ],
            }
        }
        with mock.patch.object(
            FoulPlayConfig, "pokemon_format", "gen9tugs", create=True
        ):
            battler = Battler()
            with self.assertRaises(ValueError) as captured:
                battler.initialize_first_turn_user_from_json(request_document)

        self.assertEqual(
            "Could not initialize user state from Showdown request",
            str(captured.exception),
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))


class TestSearchPrivacy(unittest.TestCase):
    def test_policy_logs_preserve_weights_without_action_labels(self):
        option = SimpleNamespace(
            move_choice=PRIVATE_SENTINEL,
            visits=20,
            total_score=10,
        )
        result = SimpleNamespace(side_one=[option], total_visits=20)

        with self.assertLogs("fp.search.main", logging.INFO) as captured:
            choice = search_main.select_move_from_mcts_results([(result, 1.0, 0)])

        output = "\n".join(captured.output)
        self.assertEqual(PRIVATE_SENTINEL, choice)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("policy weights", output)

    def test_sampled_set_logs_do_not_expose_set_or_source_details(self):
        sampled_set = SimpleNamespace(private_value=PRIVATE_SENTINEL)

        with self.assertLogs("fp.search.helpers", logging.INFO) as captured:
            search_helpers.log_pkmn_set(sampled_set, source=PRIVATE_SENTINEL)

        output = "\n".join(captured.output)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("Applied sampled Pokemon set", output)

    def test_team_preview_logs_weights_without_choice_labels(self):
        with self.assertLogs("fp.search.bss", logging.INFO) as captured:
            search_bss._log_team_preview_diagnostics({PRIVATE_SENTINEL: 0.75})

        output = "\n".join(captured.output)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("weights=[0.75]", output)

    def test_engine_move_truncation_log_omits_private_move_details(self):
        with mock.patch.object(
            FoulPlayConfig, "pokemon_format", "gen9tugs", create=True
        ):
            pkmn = Pokemon("pikachu", 100)
        pkmn.moves = [
            SimpleNamespace(
                name=f"{PRIVATE_SENTINEL}{index}",
                disabled=False,
                current_pp=1,
            )
            for index in range(5)
        ]

        with self.assertLogs(
            "fp.search.poke_engine_helpers", logging.WARNING
        ) as captured:
            poke_engine_helpers.pokemon_to_poke_engine_pkmn(pkmn)

        output = "\n".join(captured.output)
        self.assertEqual(4, len(pkmn.moves))
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("has 5 moves", output)

    def test_mcts_logs_safe_diagnostics_without_serialized_state(self):
        class State:
            @classmethod
            def from_string(cls, state):
                if state != PRIVATE_SENTINEL:
                    raise AssertionError("unexpected synthetic state")
                return object()

        with mock.patch.object(
            search_main, "PokeEngineState", State
        ), mock.patch.object(
            search_main,
            "monte_carlo_tree_search",
            return_value=SimpleNamespace(total_visits=23),
        ), self.assertLogs("fp.search.main", logging.DEBUG) as captured:
            result = search_main.get_result_from_mcts(PRIVATE_SENTINEL, 100, 3, 1)

        output = "\n".join(captured.output)
        self.assertEqual(23, result.total_visits)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("MCTS sample 3", output)
        self.assertIn("budget_ms=100", output)
        self.assertIn("Iterations 3: 23", output)

    def test_mcts_exception_and_traceback_do_not_expose_serialized_state(self):
        class RejectingState:
            @classmethod
            def from_string(cls, state):
                raise ValueError(f"invalid state {state}")

        with mock.patch.object(search_main, "PokeEngineState", RejectingState):
            with self.assertRaises(RuntimeError) as captured:
                search_main.get_result_from_mcts(PRIVATE_SENTINEL, 100, 2, 1)

        self.assertEqual(
            "Poke-engine search failed for sample 2", str(captured.exception)
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_search_serialization_exception_is_sanitized(self):
        def fail_serialization(_battle):
            raise ValueError(PRIVATE_SENTINEL)

        with mock.patch.object(
            search_main, "battle_to_poke_engine_state", fail_serialization
        ):
            with self.assertRaises(RuntimeError) as captured:
                search_main.serialize_battle_for_search(SimpleNamespace(), 5)

        self.assertEqual(
            "Could not serialize battle state for search sample 5",
            str(captured.exception),
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_damage_logging_does_not_serialize_engine_state(self):
        state = SimpleNamespace(to_string=lambda: PRIVATE_SENTINEL)
        with mock.patch.object(
            poke_engine_helpers,
            "battle_to_poke_engine_state",
            return_value=state,
        ), mock.patch.object(
            poke_engine_helpers,
            "calculate_damage",
            return_value=([10, 11], [20, 21]),
        ), self.assertLogs(
            "fp.search.poke_engine_helpers", logging.DEBUG
        ) as captured:
            rolls = poke_engine_helpers.poke_engine_get_damage_rolls(
                SimpleNamespace(battle_tag="battle-gen9tugs-safe", turn=8),
                "move-one",
                "move-two",
                True,
            )

        output = "\n".join(captured.output)
        self.assertEqual(([10, 11], [20, 21]), rolls)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("battle=battle-gen9tugs-safe", output)
        self.assertIn("turn=8", output)

    def test_damage_exception_and_traceback_are_sanitized(self):
        def fail_damage(*_args):
            raise ValueError(PRIVATE_SENTINEL)

        with mock.patch.object(
            poke_engine_helpers,
            "battle_to_poke_engine_state",
            return_value=object(),
        ), mock.patch.object(
            poke_engine_helpers, "calculate_damage", fail_damage
        ):
            with self.assertRaises(RuntimeError) as captured:
                poke_engine_helpers.poke_engine_get_damage_rolls(
                    SimpleNamespace(battle_tag="battle-safe", turn=1),
                    "move-one",
                    "move-two",
                    False,
                )

        self.assertEqual(
            "Poke-engine damage calculation failed", str(captured.exception)
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))


class TestLegacyTeamPrivacy(unittest.TestCase):
    def test_legacy_team_loading_still_returns_packed_and_structured_data(self):
        import importlib

        load_team_module = importlib.import_module("fp.teams.load_team")
        export = "Pikachu @ Light Ball\nAbility: Static\n- Thunderbolt\n"
        with mock.patch.object(
            load_team_module.os.path, "isdir", return_value=False
        ), mock.patch.object(
            load_team_module.os.path, "isfile", return_value=True
        ), mock.patch(
            "builtins.open", mock.mock_open(read_data=export)
        ):
            packed, structured, filename = load_team_module.load_team("legacy-team")

        self.assertTrue(packed)
        self.assertEqual("pikachu", structured[0]["species"])
        self.assertEqual("legacy-team", filename)

    def test_legacy_team_path_exception_does_not_expose_configured_name(self):
        import importlib

        load_team_module = importlib.import_module("fp.teams.load_team")
        with mock.patch.object(
            load_team_module.os.path, "isdir", return_value=False
        ), mock.patch.object(
            load_team_module.os.path, "isfile", return_value=False
        ):
            with self.assertRaises(ValueError) as captured:
                load_team_module.load_team(PRIVATE_SENTINEL)

        self.assertNotIn(PRIVATE_SENTINEL, str(captured.exception))
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_legacy_team_read_exception_does_not_expose_path_or_contents(self):
        import importlib

        load_team_module = importlib.import_module("fp.teams.load_team")
        with mock.patch.object(
            load_team_module.os.path, "isdir", return_value=False
        ), mock.patch.object(
            load_team_module.os.path, "isfile", return_value=True
        ), mock.patch(
            "builtins.open", side_effect=OSError(PRIVATE_SENTINEL)
        ):
            with self.assertRaises(ValueError) as captured:
                load_team_module.load_team(PRIVATE_SENTINEL)

        self.assertEqual(
            "Could not read configured team source", str(captured.exception)
        )
        self.assertNotIn(PRIVATE_SENTINEL, _formatted_traceback(captured.exception))

    def test_main_does_not_log_legacy_team_filename(self):
        socket = mock.AsyncMock()
        socket.login.return_value = "PrivacyBot"
        mode = SimpleNamespace(requires_team=True)
        configuration_values = {
            "log_level": "DEBUG",
            "log_to_file": False,
            "username": "PrivacyBot",
            "password": None,
            "websocket_uri": "ws://synthetic.invalid",
            "local_no_security_login": False,
            "avatar": None,
            "team_list": None,
            "team_name": PRIVATE_SENTINEL,
            "bot_mode": BotModes.accept_challenge,
            "pokemon_format": "gen9tugs",
            "room_name": None,
            "run_count": 1,
        }
        with ExitStack() as stack:
            for name, value in configuration_values.items():
                stack.enter_context(
                    mock.patch.object(FoulPlayConfig, name, value, create=True)
                )
            stack.enter_context(
                mock.patch.object(FoulPlayConfig, "configure", return_value=None)
            )
            stack.enter_context(mock.patch.object(main_module, "init_logging"))
            stack.enter_context(mock.patch.object(main_module, "apply_mods"))
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "load_public_prior_runtime_configuration",
                    return_value=None,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main_module.PSWebsocketClient,
                    "create",
                    new=mock.AsyncMock(return_value=socket),
                )
            )
            stack.enter_context(
                mock.patch.object(main_module, "battle_mode", return_value=mode)
            )
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "load_team",
                    return_value=(PRIVATE_SENTINEL, [], PRIVATE_SENTINEL),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "pokemon_battle",
                    new=mock.AsyncMock(return_value="PrivacyBot"),
                )
            )
            with self.assertLogs("fp.main", logging.DEBUG) as captured:
                asyncio.run(main_module.run_foul_play())

        output = "\n".join(captured.output)
        socket.update_team.assert_awaited_once_with(PRIVATE_SENTINEL)
        self.assertNotIn(PRIVATE_SENTINEL, output)
        self.assertIn("Battle won with selected team", output)


class TestDefensiveGitignore(unittest.TestCase):
    def test_rules_are_narrowly_blind_pool_specific(self):
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("**/blind_pool_private/", ignore_text)
        self.assertIn("**/blind-ladder-private/", ignore_text)
        self.assertNotIn("\nregistry.json\n", ignore_text)
        self.assertNotIn("\nstate.json\n", ignore_text)
        self.assertNotIn("\nteams/\n", ignore_text)
        self.assertNotIn("\ncanonical/\n", ignore_text)


if __name__ == "__main__":
    unittest.main()
