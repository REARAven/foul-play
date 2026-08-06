import asyncio
import copy
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

from websockets.exceptions import ConnectionClosedOK

from fp import constants
from fp.battle.protocol import activate, fieldstart, remove_item
from fp.battle.public_prior_context import PublicPriorFallback
from fp.battle.state import Battle, Move, Pokemon
from fp.config import BotModes, FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors.runtime import (
    PublicPriorConfigurationError,
    PublicPriorRuntimeConfiguration,
    PublicPriorStartupOptions,
    load_public_prior_runtime_configuration,
)
from fp.format_spec import FormatSpec
from fp.modes.standard_battle import StandardBattleMode
from fp.run_battle import start_battle
from fp.search.poke_engine_helpers import pokemon_to_poke_engine_pkmn
from fp.search.standard_battles import prepare_battles
from fp.websocket_client import (
    LocalLoginConfigurationError,
    LoginError,
    PSWebsocketClient,
    validate_loopback_websocket_uri,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TEMP_ROOT = Path(__file__).resolve().parent
ORIGINAL_MOVES = copy.deepcopy(all_move_json)
ORIGINAL_POKEDEX = copy.deepcopy(pokedex)
ORIGINAL_FORMAT = FoulPlayConfig.pokemon_format
ORIGINAL_SMOGON_STATS = FoulPlayConfig.smogon_stats


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"
    FoulPlayConfig.smogon_stats = None
    apply_mods(FormatSpec.from_format_string("gen9tugs"))


def tearDownModule():
    all_move_json.clear()
    all_move_json.update(ORIGINAL_MOVES)
    pokedex.clear()
    pokedex.update(ORIGINAL_POKEDEX)
    FoulPlayConfig.pokemon_format = ORIGINAL_FORMAT
    FoulPlayConfig.smogon_stats = ORIGINAL_SMOGON_STATS


def _variant(**overrides):
    value = {
        "variant_id": "standard",
        "weight": 1,
        "item_id": "lightball",
        "base_ability_id": "static",
        "move_ids": ["thunderbolt", "voltswitch", "surf", "protect"],
        "nature_id": "timid",
        "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
        "ivs": {"hp": 31, "atk": 0, "def": 31, "spa": 31, "spd": 31, "spe": 31},
        "level": 50,
        "source_ids": ["syntheticfixture"],
        "metadata": {"fixture": True},
    }
    value.update(overrides)
    return value


def _document(dataset_id="syntheticprior", version="1", format_id="gen9tugs", **overrides):
    value = {
        "schema_version": 1,
        "visibility": "public",
        "dataset_id": dataset_id,
        "dataset_version": version,
        "format_id": format_id,
        "patch_version": "synthetic-patch",
        "display_name": "Synthetic public prior",
        "metadata": {"fixture": True},
        "sources": [
            {
                "source_id": "syntheticfixture",
                "kind": "synthetic_test",
                "description": "Synthetic test data only",
            }
        ],
        "species": [
            {
                "species_id": "pikachu",
                "variants": [_variant()],
                "metadata": {"fixture": True},
            }
        ],
    }
    value.update(overrides)
    return value


def _private_pool_document():
    return {
        "schema_version": 1,
        "pool": {
            "pool_id": "privatepool",
            "pool_version": "1",
            "format_id": "gen9tugs",
        },
        "teams": [],
    }


def _write(directory, document, name="prior.json"):
    path = Path(directory, name)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _options(paths, fallback=PublicPriorFallback.NONE):
    return PublicPriorStartupOptions(tuple(str(path) for path in paths), fallback)


def _load_documents(*documents, fallback=PublicPriorFallback.NONE):
    with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
        paths = [
            _write(directory, document, "prior-{}.json".format(index))
            for index, document in enumerate(documents)
        ]
        return load_public_prior_runtime_configuration(
            _options(paths, fallback), "gen9tugs"
        )


def _base_argv():
    return [
        "--websocket-uri",
        "ws://synthetic.invalid",
        "--ps-username",
        "syntheticbot",
        "--bot-mode",
        "search_ladder",
        "--pokemon-format",
        "gen9tugs",
    ]


def _battle_for_datasets(context):
    return SimpleNamespace(
        public_prior_context=context,
        format_spec=FormatSpec.from_format_string("gen9tugs"),
    )


def _mode_with_dataset_mocks():
    mode = StandardBattleMode()
    mode.team_datasets = mock.Mock()
    mode.smogon_sets = mock.Mock()
    return mode


class _FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_message(self, battle_tag, messages):
        self.messages.append((battle_tag, messages))


class _FakeMode:
    def __init__(self):
        self.start_kwargs = []

    async def start_battle(self, websocket, battle_format, team_dict, **kwargs):
        self.start_kwargs.append(kwargs)
        return SimpleNamespace(battle_tag="battle-synthetic")


class TestPublicPriorCliAndLoading(unittest.TestCase):
    def test_01_no_prior_options_preserve_legacy_configuration(self):
        self.assertIsNone(FoulPlayConfig.configure(_base_argv()))

    def test_02_public_prior_file_is_repeatable(self):
        argv = _base_argv() + [
            "--public-prior-file",
            "first.json",
            "--public-prior-file",
            "second.json",
            "--public-prior-fallback",
            "none",
        ]
        self.assertEqual(("first.json", "second.json"), FoulPlayConfig.configure(argv).file_paths)

    def test_03_argument_occurrence_order_is_preserved(self):
        argv = _base_argv() + [
            "--public-prior-file",
            "z.json",
            "--public-prior-file",
            "a.json",
            "--public-prior-fallback",
            "generic",
        ]
        self.assertEqual(("z.json", "a.json"), FoulPlayConfig.configure(argv).file_paths)

    def test_04_fallback_accepts_only_generic_and_none(self):
        for value, expected in (
            ("generic", PublicPriorFallback.GENERIC),
            ("none", PublicPriorFallback.NONE),
        ):
            with self.subTest(value=value):
                options = FoulPlayConfig.configure(
                    _base_argv()
                    + ["--public-prior-file", "a.json", "--public-prior-fallback", value]
                )
                self.assertIs(expected, options.fallback_policy)
        with self.assertRaises(SystemExit):
            FoulPlayConfig.configure(
                _base_argv()
                + ["--public-prior-file", "a.json", "--public-prior-fallback", "other"]
            )

    def test_05_file_without_fallback_is_rejected(self):
        with self.assertRaises(SystemExit):
            FoulPlayConfig.configure(_base_argv() + ["--public-prior-file", "a.json"])

    def test_06_fallback_without_file_is_rejected(self):
        with self.assertRaises(SystemExit):
            FoulPlayConfig.configure(_base_argv() + ["--public-prior-fallback", "none"])

    def test_07_no_implicit_generic_fallback_is_selected(self):
        self.assertIsNone(FoulPlayConfig.configure(_base_argv()))
        with self.assertRaises(SystemExit):
            FoulPlayConfig.configure(_base_argv() + ["--public-prior-file", "a.json"])

    def test_08_local_relative_file_loads(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            _write(directory, _document(), "relative.json")
            try:
                os.chdir(directory)
                configuration = load_public_prior_runtime_configuration(
                    _options(("relative.json",)), "gen9tugs"
                )
            finally:
                os.chdir(previous)
        self.assertEqual("syntheticprior", configuration.selected_identities[0].dataset_id)

    def test_09_local_absolute_file_loads(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document()).resolve()
            configuration = load_public_prior_runtime_configuration(
                _options((path,)), "gen9tugs"
            )
        self.assertEqual(1, len(configuration.registry))

    def test_10_missing_file_fails_startup(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            missing = Path(directory, "missing.json")
            with self.assertRaisesRegex(PublicPriorConfigurationError, "does not exist"):
                load_public_prior_runtime_configuration(_options((missing,)), "gen9tugs")

    def test_11_invalid_utf8_fails_startup(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory, "invalid.json")
            path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(PublicPriorConfigurationError, "not valid UTF-8"):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")

    def test_12_malformed_json_fails_startup(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory, "malformed.json")
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(PublicPriorConfigurationError, "malformed JSON"):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")

    def test_13_wrong_schema_fails_startup(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document(schema_version=2))
            with self.assertRaisesRegex(PublicPriorConfigurationError, "schema_version"):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")

    def test_14_visibility_other_than_public_fails_startup(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document(visibility="private"))
            with self.assertRaisesRegex(PublicPriorConfigurationError, "visibility"):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")

    def test_15_private_team_pool_json_fails_public_prior_loading(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _private_pool_document())
            with self.assertRaises(PublicPriorConfigurationError):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")

    def test_16_duplicate_dataset_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            first = _write(directory, _document(), "first.json")
            second = _write(directory, _document(), "second.json")
            with self.assertRaisesRegex(PublicPriorConfigurationError, "duplicates"):
                load_public_prior_runtime_configuration(_options((first, second)), "gen9tugs")

    def test_17_duplicate_file_paths_producing_one_identity_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document())
            with self.assertRaisesRegex(PublicPriorConfigurationError, "duplicates"):
                load_public_prior_runtime_configuration(_options((path, path)), "gen9tugs")

    def test_18_format_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document(format_id="gen9ou"))
            with self.assertRaisesRegex(PublicPriorConfigurationError, "expected --pokemon-format"):
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")


class TestStartupOrdering(unittest.TestCase):
    @staticmethod
    def _options():
        return PublicPriorStartupOptions(("synthetic.json",), PublicPriorFallback.NONE)

    def test_19_validation_occurs_after_format_overlay(self):
        import fp.main as main_module

        events = []
        with mock.patch.object(FoulPlayConfig, "configure", side_effect=lambda: events.append("configure") or self._options()), mock.patch.object(
            main_module, "init_logging"
        ), mock.patch.object(main_module, "apply_mods", side_effect=lambda spec: events.append("overlay")), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            side_effect=lambda options, format_id: events.append("validate") or (_ for _ in ()).throw(PublicPriorConfigurationError("stop")),
        ), mock.patch.object(main_module.PSWebsocketClient, "create", new=mock.AsyncMock()) as create:
            with self.assertRaises(PublicPriorConfigurationError):
                asyncio.run(main_module.run_foul_play())
        self.assertEqual(["configure", "overlay", "validate"], events)
        create.assert_not_awaited()

    def test_20_validation_occurs_before_websocket_connection(self):
        import fp.main as main_module

        events = []
        sentinel = RuntimeError("connected")
        with mock.patch.object(FoulPlayConfig, "configure", return_value=self._options()), mock.patch.object(
            main_module, "init_logging"
        ), mock.patch.object(main_module, "apply_mods", side_effect=lambda spec: events.append("overlay")), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            side_effect=lambda options, format_id: events.append("validate") or mock.sentinel.configuration,
        ), mock.patch.object(
            main_module.PSWebsocketClient,
            "create",
            new=mock.AsyncMock(side_effect=lambda *args: events.append("connect") or (_ for _ in ()).throw(sentinel)),
        ):
            with self.assertRaisesRegex(RuntimeError, "connected"):
                asyncio.run(main_module.run_foul_play())
        self.assertEqual(["overlay", "validate", "connect"], events)

    def test_21_validation_failure_performs_no_login_attempt(self):
        import fp.main as main_module

        with mock.patch.object(FoulPlayConfig, "configure", return_value=self._options()), mock.patch.object(
            main_module, "init_logging"
        ), mock.patch.object(main_module, "apply_mods"), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            side_effect=PublicPriorConfigurationError("invalid prior"),
        ), mock.patch.object(main_module.PSWebsocketClient, "create", new=mock.AsyncMock()) as create:
            with self.assertRaises(PublicPriorConfigurationError):
                asyncio.run(main_module.run_foul_play())
        create.assert_not_awaited()


class TestRuntimeConfiguration(unittest.TestCase):
    def test_22_loaded_registry_is_immutable(self):
        configuration = _load_documents(_document())
        self.assertIsInstance(configuration.registry.dataset_lookup, MappingProxyType)
        with self.assertRaises(TypeError):
            configuration.registry.dataset_lookup[configuration.selected_identities[0]] = None

    def test_23_selected_identity_order_matches_file_order(self):
        configuration = _load_documents(
            _document("zetaprior"), _document("alphaprior")
        )
        self.assertEqual(
            ("zetaprior", "alphaprior"),
            tuple(identity.dataset_id for identity in configuration.selected_identities),
        )

    def test_24_runtime_configuration_is_immutable(self):
        configuration = _load_documents(_document())
        with self.assertRaises(FrozenInstanceError):
            configuration.format_id = "gen9ou"

    def test_25_runtime_configuration_creates_fresh_battle_contexts(self):
        configuration = _load_documents(_document())
        self.assertIsNot(
            configuration.create_battle_context("gen9tugs"),
            configuration.create_battle_context("gen9tugs"),
        )

    def test_26_two_battles_do_not_share_observations(self):
        configuration = _load_documents(_document())
        first = Battle("first", public_prior_context=configuration.create_battle_context("gen9tugs"))
        second = Battle("second", public_prior_context=configuration.create_battle_context("gen9tugs"))
        first.team_inference.record_public_member("pikachu", 50)
        self.assertEqual(1, len(first.team_inference.observation_ledger.members))
        self.assertEqual(0, len(second.team_inference.observation_ledger.members))

    def test_27_immutable_datasets_may_be_shared_safely(self):
        configuration = _load_documents(_document())
        first = configuration.create_battle_context("gen9tugs")
        second = configuration.create_battle_context("gen9tugs")
        self.assertIs(first.registry, second.registry)
        self.assertIs(
            first.registry.get(first.selected_identities[0]),
            second.registry.get(second.selected_identities[0]),
        )

    def test_28_registry_is_not_stored_on_standard_battle_mode(self):
        mode = StandardBattleMode()
        self.assertFalse(hasattr(mode, "public_prior_configuration"))
        self.assertFalse(any(type(value).__name__ == "PublicPriorRegistry" for value in vars(mode).values()))

    def test_29_no_context_is_created_without_prior_files(self):
        mode = _FakeMode()
        socket = _FakeSocket()
        with mock.patch("fp.run_battle.battle_mode", return_value=mode):
            asyncio.run(start_battle(socket, "gen9tugs", None))
        self.assertNotIn("public_prior_context", mode.start_kwargs[0])

    def test_30_context_fallback_is_generic_when_explicitly_configured(self):
        configuration = _load_documents(_document(), fallback=PublicPriorFallback.GENERIC)
        self.assertIs(
            PublicPriorFallback.GENERIC,
            configuration.create_battle_context("gen9tugs").fallback_policy,
        )

    def test_31_context_fallback_is_none_when_explicitly_configured(self):
        configuration = _load_documents(_document(), fallback=PublicPriorFallback.NONE)
        self.assertIs(
            PublicPriorFallback.NONE,
            configuration.create_battle_context("gen9tugs").fallback_policy,
        )

    def test_32_battle_format_matches_context_format(self):
        configuration = _load_documents(_document())
        self.assertEqual("gen9tugs", configuration.create_battle_context("gen9tugs").format_id)
        with self.assertRaises(PublicPriorConfigurationError):
            configuration.create_battle_context("gen9ou")

    def test_33_sequential_battles_receive_fresh_context_objects(self):
        configuration = _load_documents(_document())
        mode = _FakeMode()
        socket = _FakeSocket()
        with mock.patch("fp.run_battle.battle_mode", return_value=mode):
            asyncio.run(start_battle(socket, "gen9tugs", None, public_prior_configuration=configuration))
            asyncio.run(start_battle(socket, "gen9tugs", None, public_prior_configuration=configuration))
        self.assertIsNot(
            mode.start_kwargs[0]["public_prior_context"],
            mode.start_kwargs[1]["public_prior_context"],
        )

    def test_34_public_files_are_loaded_once_not_once_per_battle(self):
        import fp.data.public_priors.runtime as runtime

        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document())
            with mock.patch(
                "fp.data.public_priors.runtime.load_public_prior",
                wraps=runtime.load_public_prior,
            ) as loader:
                configuration = load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")
            mode = _FakeMode()
            socket = _FakeSocket()
            with mock.patch("fp.run_battle.battle_mode", return_value=mode):
                asyncio.run(start_battle(socket, "gen9tugs", None, public_prior_configuration=configuration))
                asyncio.run(start_battle(socket, "gen9tugs", None, public_prior_configuration=configuration))
        self.assertEqual(1, loader.call_count)

    def test_35_no_directory_scanning_occurs(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory, mock.patch.object(
            Path, "glob", side_effect=AssertionError("glob called")
        ), mock.patch.object(Path, "rglob", side_effect=AssertionError("rglob called")), mock.patch(
            "os.walk", side_effect=AssertionError("walk called")
        ):
            path = _write(directory, _document())
            self.assertEqual(1, len(load_public_prior_runtime_configuration(_options((path,)), "gen9tugs").registry))

    def test_36_no_glob_expansion_occurs_inside_foul_play(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            _write(directory, _document())
            wildcard = Path(directory, "*.json")
            with self.assertRaises(PublicPriorConfigurationError):
                load_public_prior_runtime_configuration(_options((wildcard,)), "gen9tugs")

    def test_37_no_url_input_is_supported(self):
        with mock.patch("fp.data.public_priors.runtime.load_public_prior") as loader:
            with self.assertRaisesRegex(PublicPriorConfigurationError, "local filesystem"):
                load_public_prior_runtime_configuration(
                    _options(("https://example.invalid/prior.json",)), "gen9tugs"
                )
        loader.assert_not_called()

    def test_38_startup_creates_no_public_prior_cache(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document())
            before = tuple(sorted(item.name for item in Path(directory).iterdir()))
            load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")
            after = tuple(sorted(item.name for item in Path(directory).iterdir()))
        self.assertEqual(before, after)

    def test_39_source_prior_files_remain_unchanged(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document())
            before = path.read_bytes()
            load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")
            after = path.read_bytes()
        self.assertEqual(before, after)

    def test_40_safe_logging_includes_dataset_identity(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, _document())
            with self.assertLogs("fp.data.public_priors.runtime", level="INFO") as captured:
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")
        text = "\n".join(captured.output)
        self.assertIn("dataset_id=syntheticprior", text)
        self.assertIn("dataset_version=1", text)
        self.assertIn("precedence 1", text)

    def test_41_safe_logging_excludes_full_variants(self):
        document = _document()
        document["species"][0]["variants"][0]["metadata"] = {"secretmarker": "neverlogthis"}
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write(directory, document)
            with self.assertLogs("fp.data.public_priors.runtime", level="INFO") as captured:
                load_public_prior_runtime_configuration(_options((path,)), "gen9tugs")
        text = "\n".join(captured.output)
        self.assertNotIn("neverlogthis", text)
        self.assertNotIn("thunderbolt", text)


class TestGenericDatasetPolicy(unittest.TestCase):
    def _assert_initializes(self, context, *, preview=True):
        mode = _mode_with_dataset_mocks()
        message = "preview" if preview else None
        result = mode.initialize_datasets_if_enabled(
            _battle_for_datasets(context),
            "gen9tugs",
            {"pikachu"},
            team_preview_message=message,
        )
        return mode, result

    def test_42_no_context_preview_initializes_team_datasets(self):
        mode, result = self._assert_initializes(None)
        self.assertTrue(result)
        mode.team_datasets.initialize.assert_called_once()

    def test_43_no_context_preview_initializes_smogon_sets(self):
        mode, _ = self._assert_initializes(None)
        mode.smogon_sets.initialize.assert_called_once()

    def test_44_generic_preview_initializes_team_datasets(self):
        context = _load_documents(_document(), fallback=PublicPriorFallback.GENERIC).create_battle_context("gen9tugs")
        mode, _ = self._assert_initializes(context)
        mode.team_datasets.initialize.assert_called_once()

    def test_45_generic_preview_initializes_smogon_sets(self):
        context = _load_documents(_document(), fallback=PublicPriorFallback.GENERIC).create_battle_context("gen9tugs")
        mode, _ = self._assert_initializes(context)
        mode.smogon_sets.initialize.assert_called_once()

    def test_46_none_preview_does_not_initialize_team_datasets(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        for preview in (True, False):
            with self.subTest(preview=preview):
                mode, result = self._assert_initializes(context, preview=preview)
                self.assertFalse(result)
                mode.team_datasets.initialize.assert_not_called()

    def test_47_none_preview_does_not_initialize_smogon_sets(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        for preview in (True, False):
            with self.subTest(preview=preview):
                mode, _ = self._assert_initializes(context, preview=preview)
                mode.smogon_sets.initialize.assert_not_called()

    def test_48_none_performs_no_team_datasets_cache_read(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        with mock.patch("builtins.open", side_effect=AssertionError("cache read")) as opened:
            self._assert_initializes(context)
        opened.assert_not_called()

    def test_49_none_performs_no_smogon_cache_read(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        with mock.patch("fp.data.sets.smogon.json.load", side_effect=AssertionError("cache read")) as loaded:
            self._assert_initializes(context)
        loaded.assert_not_called()

    def test_50_none_performs_no_team_datasets_network_call(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        with mock.patch("fp.data.sets.base.requests.get", side_effect=AssertionError("network")) as requested:
            self._assert_initializes(context)
        requested.assert_not_called()

    def test_51_none_performs_no_smogon_network_call(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        with mock.patch("fp.data.sets.smogon.requests.get", side_effect=AssertionError("network")) as requested:
            self._assert_initializes(context)
        requested.assert_not_called()

    def test_52_none_creates_no_generic_cache_directory(self):
        context = _load_documents(_document()).create_battle_context("gen9tugs")
        with mock.patch("os.makedirs", side_effect=AssertionError("cache directory")) as makedirs:
            self._assert_initializes(context)
        makedirs.assert_not_called()
        script = (
            "from unittest import mock; "
            "guard = mock.patch('os.makedirs', side_effect=AssertionError('cache directory')); "
            "patched = guard.start(); "
            "import fp.data.sets.base, fp.data.sets.smogon; "
            "assert not patched.called; "
            "guard.stop()"
        )
        subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            check=True,
        )

    def test_53_none_public_miss_leaves_pokemon_unsampled(self):
        from test_public_prior_sampling import _battle, _context, _dataset

        context = _context(_dataset(species_id="raichu"), fallback=PublicPriorFallback.NONE)
        battle = _battle(context)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        self.assertEqual(0, len(sampled.opponent.active.moves))

    def test_54_generic_public_miss_invokes_existing_generic_sampler(self):
        from test_public_prior_sampling import _battle, _context, _dataset

        context = _context(_dataset(species_id="raichu"), fallback=PublicPriorFallback.GENERIC)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            prepare_battles(_battle(context), 1)
        generic.assert_called_once()

    def test_55_public_success_bypasses_generic_sampling(self):
        from test_public_prior_sampling import _battle, _context, _dataset

        context = _context(_dataset(), fallback=PublicPriorFallback.GENERIC)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(_battle(context), 1)[0][0]
        generic.assert_not_called()
        self.assertEqual(4, len(sampled.opponent.active.moves))

    def test_56_smogon_stats_format_with_none_does_not_initialize_smogon(self):
        previous = FoulPlayConfig.smogon_stats
        FoulPlayConfig.smogon_stats = "gen9ou"
        try:
            context = _load_documents(_document()).create_battle_context("gen9tugs")
            mode, _ = self._assert_initializes(context)
        finally:
            FoulPlayConfig.smogon_stats = previous
        mode.smogon_sets.initialize.assert_not_called()

    def test_57_smogon_stats_format_with_generic_retains_current_behavior(self):
        previous = FoulPlayConfig.smogon_stats
        FoulPlayConfig.smogon_stats = "gen9ou"
        try:
            context = _load_documents(_document(), fallback=PublicPriorFallback.GENERIC).create_battle_context("gen9tugs")
            mode, _ = self._assert_initializes(context)
        finally:
            FoulPlayConfig.smogon_stats = previous
        format_spec = mode.smogon_sets.initialize.call_args.args[0]
        self.assertEqual("gen9ou", format_spec.full_name)

    def test_58_ordinary_gen9_is_unchanged_without_prior_configuration(self):
        mode = _mode_with_dataset_mocks()
        battle = SimpleNamespace(
            public_prior_context=None,
            format_spec=FormatSpec.from_format_string("gen9ou"),
        )
        self.assertTrue(mode.initialize_datasets_if_enabled(battle, "gen9ou", {"pikachu"}))
        mode.team_datasets.initialize.assert_called_once()
        mode.smogon_sets.initialize.assert_called_once()

    def test_59_national_dex_is_unchanged_without_prior_configuration(self):
        mode = _mode_with_dataset_mocks()
        battle = SimpleNamespace(
            public_prior_context=None,
            format_spec=FormatSpec.from_format_string("gen9nationaldex"),
        )
        self.assertTrue(mode.initialize_datasets_if_enabled(battle, "gen9nationaldex", {"pikachu"}))
        mode.team_datasets.initialize.assert_called_once()
        mode.smogon_sets.initialize.assert_called_once()

    def test_60_tugs_public_prior_does_not_apply_to_another_format(self):
        configuration = _load_documents(_document())
        with self.assertRaisesRegex(PublicPriorConfigurationError, "does not match"):
            configuration.create_battle_context("gen9ou")


class TestFirewallRegressionsAndMechanics(unittest.TestCase):
    def test_61_startup_configuration_imports_no_private_team_pool_model(self):
        import fp.data.public_priors.runtime as runtime

        source = inspect.getsource(runtime)
        self.assertNotIn("team_pools", source)
        for name in ("TeamPool", "TeamRecord", "PokemonRecord"):
            self.assertNotIn(name, source)

    def test_62_runtime_search_imports_no_private_team_pool_model(self):
        import fp.search.public_prior_sampling as sampling
        import fp.search.standard_battles as standard

        source = inspect.getsource(sampling) + inspect.getsource(standard)
        self.assertNotIn("fp.data.team_pools", source)
        self.assertNotIn("TeamPoolRegistry", source)

    def test_63_no_private_candidate_id_influences_public_configuration(self):
        import fp.battle.public_prior_context as context
        import fp.data.public_priors.runtime as runtime

        source = inspect.getsource(context) + inspect.getsource(runtime)
        for name in ("TeamPoolCandidateId", "candidate_ids", "roster_key"):
            self.assertNotIn(name, source)

    def _focused_count(self, pattern):
        suite = unittest.TestLoader().discover(str(ROOT / "tests"), pattern=pattern)
        return suite.countTestCases()

    def test_64_existing_phase_one_tests_remain_present(self):
        self.assertEqual(55, self._focused_count("test_team_pool_loader.py"))

    def test_65_existing_phase_two_tests_remain_present(self):
        self.assertEqual(49, self._focused_count("test_team_pool_inference.py"))

    def test_66_existing_phase_three_tests_remain_present(self):
        self.assertEqual(113, self._focused_count("test_team_pool_observation_filtering.py"))

    def test_67_existing_phase_four_tests_remain_present(self):
        self.assertEqual(80, self._focused_count("test_public_prior_loader.py"))

    def test_68_existing_phase_five_tests_remain_present(self):
        self.assertEqual(97, self._focused_count("test_public_prior_sampling.py"))

    def test_69_existing_crag_mend_behavior_passes(self):
        self.assertEqual(5, all_move_json["cragmend"][constants.PP])
        self.assertEqual([1, 2], all_move_json["cragmend"]["heal"])
        self.assertEqual(8, Move("cragmend").max_pp)

    def test_70_existing_trick_room_persistent_behavior_passes(self):
        battle = Battle("synthetic")
        battle.generation = "gen9"
        battle.pokemon_format = "gen9tugs"
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.user.active = Pokemon("weedle", 100)
        battle.opponent.active = Pokemon("caterpie", 100)
        battle.user.active.ability = "persistent"
        fieldstart(
            battle,
            ["", "-fieldstart", "move: Trick Room", "[of] p1a: Weedle", "[persistent]"],
        )
        self.assertTrue(battle.trick_room)
        self.assertEqual(8, battle.trick_room_turns_remaining)

    def test_71_existing_ancient_shell_serialization_passes(self):
        pokemon = Pokemon("bastiodon", 100)
        pokemon.ability = "ancientshell"
        self.assertEqual("ancientshell", pokemon_to_poke_engine_pkmn(pokemon).ability)

    def test_72_existing_corrosive_gas_parsing_passes(self):
        battle = Battle("synthetic")
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.opponent.active = Pokemon("dustox", 100)
        battle.opponent.active.item = "leftovers"
        remove_item(
            battle,
            ["", "-enditem", "p2a: Dustox", "Leftovers", "[from] move: Corrosive Gas"],
        )
        self.assertEqual("leftovers", battle.opponent.active.removed_item)

    def test_73_existing_closing_jaws_bookkeeping_passes(self):
        battle = Battle("synthetic")
        battle.user.name = "p1"
        battle.opponent.name = "p2"
        battle.opponent.active = Pokemon("mawile", 100)
        activate(battle, ["", "-activate", "p2a: Mawile", "ability: Closing Jaws"])
        self.assertEqual("closingjaws", battle.opponent.active.ability)

    def test_74_context_reaches_standard_mode_battle_construction(self):
        configuration = _load_documents(_document())
        context = configuration.create_battle_context("gen9tugs")

        class LifecycleMode(StandardBattleMode):
            async def start_battle_common(
                self,
                websocket,
                battle_format,
                team_sheet_policy=None,
                public_prior_context=None,
            ):
                battle = Battle(
                    "battle-synthetic",
                    team_sheet_policy=team_sheet_policy,
                    public_prior_context=public_prior_context,
                )
                battle.pokemon_format = battle_format
                battle.generation = "gen9"
                battle.user.name = "p1"
                battle.opponent.name = "p2"
                battle.user.active = Pokemon("weedle", 50)
                return battle, "|clearpoke|\n|poke|p2|Pikachu, L50"

            async def handle_team_preview(self, battle, websocket):
                return None

        mode = LifecycleMode()
        with mock.patch(
            "fp.modes.standard_battle.get_first_request_json",
            new=mock.AsyncMock(),
        ):
            battle = asyncio.run(
                mode.start_battle(
                    _FakeSocket(),
                    "gen9tugs",
                    None,
                    public_prior_context=context,
                )
            )
        self.assertIs(context, battle.public_prior_context)
        mode.team_datasets.initialize = mock.Mock()
        mode.smogon_sets.initialize = mock.Mock()
        self.assertFalse(
            mode.initialize_datasets_if_enabled(
                battle, "gen9tugs", {"pikachu"}
            )
        )

    def test_75_main_loads_once_and_injects_same_factory_for_two_battles(self):
        import fp.main as main_module

        configuration = _load_documents(_document())
        socket = mock.AsyncMock()
        socket.login.return_value = "syntheticbot"
        mode = SimpleNamespace(requires_team=False)
        FoulPlayConfig.log_level = "INFO"
        FoulPlayConfig.log_to_file = False
        FoulPlayConfig.username = "syntheticbot"
        FoulPlayConfig.password = None
        FoulPlayConfig.websocket_uri = "ws://synthetic.invalid"
        FoulPlayConfig.avatar = None
        FoulPlayConfig.team_list = None
        FoulPlayConfig.bot_mode = BotModes.search_ladder
        FoulPlayConfig.pokemon_format = "gen9tugs"
        FoulPlayConfig.run_count = 2
        with mock.patch.object(
            FoulPlayConfig,
            "configure",
            return_value=PublicPriorStartupOptions(
                ("synthetic.json",), PublicPriorFallback.NONE
            ),
        ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
            main_module, "apply_mods"
        ), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            return_value=configuration,
        ) as loader, mock.patch.object(
            main_module.PSWebsocketClient,
            "create",
            new=mock.AsyncMock(return_value=socket),
        ), mock.patch.object(
            main_module, "battle_mode", return_value=mode
        ), mock.patch.object(
            main_module,
            "pokemon_battle",
            new=mock.AsyncMock(return_value="syntheticbot"),
        ) as battle_runner:
            asyncio.run(main_module.run_foul_play())
        loader.assert_called_once()
        self.assertEqual(2, battle_runner.await_count)
        for call in battle_runner.await_args_list:
            self.assertIs(
                configuration,
                call.kwargs["public_prior_configuration"],
            )


LOCAL_LOGIN_BASE_ARGV = [
    "--websocket-uri",
    "ws://synthetic.invalid/showdown/websocket",
    "--ps-username",
    "LocalBot",
    "--bot-mode",
    "search_ladder",
    "--pokemon-format",
    "gen9tugs",
]


class _LocalLoginFakeWebsocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def recv(self):
        next_message = self.messages.pop(0)
        if isinstance(next_message, BaseException):
            raise next_message
        return next_message

    async def send(self, message):
        self.sent.append(message)


def _local_login_client(messages, *, local=False, password=None):
    client = PSWebsocketClient()
    client.username = "LocalBot"
    client.password = password
    client.address = "ws://127.0.0.1:8013/showdown/websocket"
    client.local_no_security_login = local
    client.login_uri = (
        "https://play.pokemonshowdown.com/api/login"
        if password
        else "https://play.pokemonshowdown.com/action.php?"
    )
    client.websocket = _LocalLoginFakeWebsocket(messages)
    return client


class TestZLocalNoSecurityConfiguration(unittest.TestCase):
    def tearDown(self):
        FoulPlayConfig.local_no_security_login = False
        FoulPlayConfig.password = None

    def test_76_default_mode_is_not_inferred_for_a_loopback_uri(self):
        argv = list(LOCAL_LOGIN_BASE_ARGV)
        argv[1] = "ws://127.0.0.1:8013/showdown/websocket"
        FoulPlayConfig.configure(argv)
        self.assertFalse(FoulPlayConfig.local_no_security_login)

    def test_77_explicit_cli_option_enables_local_mode(self):
        argv = list(LOCAL_LOGIN_BASE_ARGV)
        argv[1] = "ws://localhost:8013/showdown/websocket"
        FoulPlayConfig.configure(argv + ["--local-no-security-login"])
        self.assertTrue(FoulPlayConfig.local_no_security_login)

    def test_78_loopback_forms_are_accepted(self):
        for uri, expected in (
            ("ws://127.0.0.1:8013/showdown/websocket", "127.0.0.1"),
            ("ws://localhost:8013/showdown/websocket", "localhost"),
            ("ws://[::1]:8013/showdown/websocket", "::1"),
        ):
            with self.subTest(uri=uri):
                self.assertEqual(expected, validate_loopback_websocket_uri(uri))

    def test_79_non_loopback_destinations_are_rejected(self):
        for uri in (
            "wss://play.pokemonshowdown.com/showdown/websocket",
            "ws://192.168.1.2:8013/showdown/websocket",
            "ws://10.0.0.2:8013/showdown/websocket",
            "ws://0.0.0.0:8013/showdown/websocket",
            "ws://localhost.example:8013/showdown/websocket",
        ):
            with self.subTest(uri=uri), self.assertRaises(
                LocalLoginConfigurationError
            ):
                validate_loopback_websocket_uri(uri)

    def test_80_malformed_or_ambiguous_uris_are_rejected(self):
        for uri in (
            "127.0.0.1:8013",
            "http://127.0.0.1:8013",
            "ws://[::1",
            "ws://user@127.0.0.1:8013/showdown/websocket",
            "ws://127.0.0.1:notaport/showdown/websocket",
        ):
            with self.subTest(uri=uri), self.assertRaises(
                LocalLoginConfigurationError
            ):
                validate_loopback_websocket_uri(uri)

    def test_81_password_conflict_is_rejected_by_cli(self):
        argv = list(LOCAL_LOGIN_BASE_ARGV)
        argv[1] = "ws://127.0.0.1:8013/showdown/websocket"
        with self.assertRaises(SystemExit):
            FoulPlayConfig.configure(
                argv
                + [
                    "--local-no-security-login",
                    "--ps-password",
                    "unused-test-password",
                ]
            )

    def test_82_direct_client_creation_rejects_non_loopback_before_connect(self):
        with mock.patch(
            "fp.websocket_client.websockets.connect", new=mock.AsyncMock()
        ) as connect:
            with self.assertRaises(LocalLoginConfigurationError):
                asyncio.run(
                    PSWebsocketClient.create(
                        "LocalBot",
                        None,
                        "ws://192.168.1.2:8013/showdown/websocket",
                        local_no_security_login=True,
                    )
                )
        connect.assert_not_awaited()

    def test_83_direct_password_conflict_fails_before_connecting(self):
        with mock.patch(
            "fp.websocket_client.websockets.connect", new=mock.AsyncMock()
        ) as connect:
            with self.assertRaises(LocalLoginConfigurationError):
                asyncio.run(
                    PSWebsocketClient.create(
                        "LocalBot",
                        "unused-test-password",
                        "ws://127.0.0.1:8013/showdown/websocket",
                        local_no_security_login=True,
                    )
                )
        connect.assert_not_awaited()


class TestZLocalNoSecurityProtocols(unittest.IsolatedAsyncioTestCase):
    async def test_84_default_mode_preserves_public_assertion_flow(self):
        client = _local_login_client(["|challstr|4|challenge-value"])
        response = SimpleNamespace(
            status_code=200, text="assertion-value", content=b""
        )
        with mock.patch(
            "fp.websocket_client.requests.post", return_value=response
        ) as post, mock.patch(
            "fp.websocket_client.asyncio.sleep", new=mock.AsyncMock()
        ) as sleep:
            user_id = await client.login()

        post.assert_called_once_with(
            "https://play.pokemonshowdown.com/action.php?",
            data={
                "act": "getassertion",
                "userid": "LocalBot",
                "challstr": "4|challenge-value",
            },
        )
        self.assertEqual(["|/trn LocalBot,0,assertion-value"], client.websocket.sent)
        sleep.assert_awaited_once_with(3)
        self.assertEqual("LocalBot", user_id)
        self.assertIsNone(client.last_message)

    async def test_85_registered_password_flow_is_unchanged(self):
        client = _local_login_client(
            ["|challstr|4|challenge-value"], password="password-value"
        )
        response = SimpleNamespace(
            status_code=200,
            text=(
                ']{"actionsuccess":true,"assertion":"assertion-value",'
                '"curuser":{"userid":"localbot"}}'
            ),
            content=b"",
        )
        with mock.patch(
            "fp.websocket_client.requests.post", return_value=response
        ) as post, mock.patch(
            "fp.websocket_client.asyncio.sleep", new=mock.AsyncMock()
        ) as sleep:
            user_id = await client.login()

        post.assert_called_once_with(
            "https://play.pokemonshowdown.com/api/login",
            data={
                "name": "LocalBot",
                "pass": "password-value",
                "challstr": "4|challenge-value",
            },
        )
        self.assertEqual(["|/trn LocalBot,0,assertion-value"], client.websocket.sent)
        sleep.assert_awaited_once_with(3)
        self.assertEqual("localbot", user_id)
        self.assertIsNone(client.last_message)

    async def test_86_local_mode_uses_empty_token_and_no_http(self):
        client = _local_login_client(
            [
                "|challstr|4|challenge-value",
                '|updateuser| LocalBot|1|1|{"blockChallenges":false}',
            ],
            local=True,
        )
        with mock.patch(
            "fp.websocket_client.requests.get",
            side_effect=AssertionError("HTTP GET must not be called"),
        ) as get, mock.patch(
            "fp.websocket_client.requests.post",
            side_effect=AssertionError("HTTP POST must not be called"),
        ) as post:
            user_id = await client.login()

        get.assert_not_called()
        post.assert_not_called()
        self.assertEqual(["|/trn LocalBot,0,"], client.websocket.sent)
        self.assertEqual("LocalBot", user_id)
        self.assertIsNone(client.last_message)

    async def test_87_local_authentication_values_are_absent_from_logs(self):
        client = _local_login_client(
            [
                "|challstr|4|challenge-value",
                '|updateuser| LocalBot|1|1|{"blockChallenges":false}',
            ],
            local=True,
        )
        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            await client.login()
        output = "\n".join(captured.output)
        self.assertNotIn("challenge-value", output)
        self.assertNotIn("|/trn LocalBot,0,", output)
        self.assertNotIn("assertion-value", output)
        self.assertIn("Local username claim succeeded", output)

    async def test_88_public_assertion_is_absent_from_logs(self):
        client = _local_login_client(["|challstr|4|challenge-value"])
        response = SimpleNamespace(
            status_code=200, text="assertion-value", content=b""
        )
        with mock.patch(
            "fp.websocket_client.requests.post", return_value=response
        ), mock.patch(
            "fp.websocket_client.asyncio.sleep", new=mock.AsyncMock()
        ), self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            await client.login()
        output = "\n".join(captured.output)
        self.assertNotIn("challenge-value", output)
        self.assertNotIn("assertion-value", output)
        self.assertNotIn("|/trn LocalBot,0,", output)

    async def test_89_local_rejection_is_sanitized(self):
        client = _local_login_client(
            [
                "|challstr|4|challenge-value",
                "|nametaken|LocalBot|rejected-detail",
            ],
            local=True,
        )
        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            with self.assertRaisesRegex(
                LoginError, "Local username claim was rejected"
            ):
                await client.login()
        output = "\n".join(captured.output)
        self.assertNotIn("challenge-value", output)
        self.assertNotIn("rejected-detail", output)

    async def test_90_connection_close_before_confirmation_is_sanitized(self):
        client = _local_login_client(
            [
                "|challstr|4|challenge-value",
                ConnectionClosedOK(None, None),
            ],
            local=True,
        )
        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            with self.assertRaisesRegex(LoginError, "Connection closed before"):
                await client.login()
        self.assertNotIn("challenge-value", "\n".join(captured.output))


class TestZLocalNoSecurityWiring(unittest.TestCase):
    def test_91_main_passes_explicit_local_mode_to_client(self):
        import fp.main as main_module

        previous = FoulPlayConfig.local_no_security_login
        FoulPlayConfig.log_level = "INFO"
        FoulPlayConfig.log_to_file = False
        FoulPlayConfig.username = "LocalBot"
        FoulPlayConfig.password = None
        FoulPlayConfig.websocket_uri = "ws://127.0.0.1:8013/showdown/websocket"
        FoulPlayConfig.local_no_security_login = True
        FoulPlayConfig.pokemon_format = "gen9tugs"
        try:
            with mock.patch.object(
                FoulPlayConfig, "configure", return_value=None
            ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
                main_module, "apply_mods"
            ), mock.patch.object(
                main_module,
                "load_public_prior_runtime_configuration",
                return_value=None,
            ), mock.patch.object(
                main_module.PSWebsocketClient,
                "create",
                new=mock.AsyncMock(side_effect=RuntimeError("stop after create")),
            ) as create:
                with self.assertRaisesRegex(RuntimeError, "stop after create"):
                    asyncio.run(main_module.run_foul_play())
            create.assert_awaited_once_with(
                "LocalBot",
                None,
                "ws://127.0.0.1:8013/showdown/websocket",
                True,
            )
        finally:
            FoulPlayConfig.local_no_security_login = previous


if __name__ == "__main__":
    unittest.main()
