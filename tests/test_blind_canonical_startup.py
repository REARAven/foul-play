from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
import importlib
import io
import json
import logging
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from fp.config import (
    BotModes,
    FoulPlayConfig,
    TEAM_SOURCE_BLIND_CANONICAL,
    TEAM_SOURCE_LEGACY,
)
from fp.data.blind_pool.activation import run_blind_canonical_activation
from fp.data.blind_pool.bag import BlindPoolBagStore
from fp.data.blind_pool.errors import (
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
)
from fp.data.blind_pool.models import (
    BlindChallengeToken,
    BlindPoolConfig,
    BlindPoolStateConfig,
)
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.startup import (
    CANONICAL_REGISTRY_PATH_ENV,
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    BlindCanonicalStartupConfig,
    load_blind_canonical_startup_config,
    prepare_blind_canonical_deployment,
)
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture
from tests.test_blind_pool_lifecycle import (
    EXACT_CHALLENGE,
    FakeTransport,
    exact_room_message,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SENTINEL = "PHASE5E1-SYNTHETIC-PRIVATE-SENTINEL"
TOKEN = "a" * 32


def _base_argv(*extra: str) -> list[str]:
    return [
        "--websocket-uri",
        "ws://synthetic.invalid",
        "--ps-username",
        "SyntheticBot",
        "--bot-mode",
        "accept_challenge",
        "--pokemon-format",
        "gen9tugs",
        *extra,
    ]


class TestBlindCanonicalConfiguration(unittest.TestCase):
    def config(self):
        return type(FoulPlayConfig)()

    def test_default_team_source_is_unconditional_legacy(self):
        config = self.config()
        with mock.patch.dict(
            os.environ,
            {
                "TUGS_BLIND_POOL_ROOT": PRIVATE_SENTINEL,
                "TUGS_BLIND_POOL_REGISTRY": PRIVATE_SENTINEL,
                "TUGS_BLIND_POOL_STATE": PRIVATE_SENTINEL,
                CANONICAL_REGISTRY_PATH_ENV: PRIVATE_SENTINEL,
            },
            clear=False,
        ):
            config.configure(_base_argv())
        self.assertEqual(TEAM_SOURCE_LEGACY, config.team_source)
        self.assertEqual("gen9tugs", config.team_name)

    def test_explicit_canonical_source_is_accepted(self):
        config = self.config()
        config.configure(_base_argv("--team-source", "blind-canonical"))
        self.assertEqual(TEAM_SOURCE_BLIND_CANONICAL, config.team_source)
        self.assertIsNone(config.team_name)
        self.assertIsNone(config.team_list)

    def test_environment_presence_does_not_activate_canonical_source(self):
        config = self.config()
        config.configure(_base_argv())
        self.assertEqual(TEAM_SOURCE_LEGACY, config.team_source)

    def test_canonical_conflicts_fail_without_filesystem_or_network(self):
        cases = (
            ("team-name", ("--team-name", PRIVATE_SENTINEL)),
            ("team-list", ("--team-list", PRIVATE_SENTINEL)),
            ("room-name", ("--room-name", PRIVATE_SENTINEL)),
            ("user", ("--user-to-challenge", PRIVATE_SENTINEL)),
            (
                "format",
                (
                    "--team-source",
                    "blind-canonical",
                    "--pokemon-format",
                    "gen9ou",
                ),
            ),
            (
                "mode",
                (
                    "--team-source",
                    "blind-canonical",
                    "--bot-mode",
                    "search_ladder",
                ),
            ),
        )
        for name, extra in cases:
            argv = _base_argv("--team-source", "blind-canonical", *extra)
            errors = io.StringIO()
            with self.subTest(name=name), redirect_stderr(errors):
                with mock.patch(
                    "pathlib.Path.resolve",
                    side_effect=AssertionError("filesystem touched"),
                ), self.assertRaises(SystemExit):
                    self.config().configure(argv)
            self.assertNotIn(PRIVATE_SENTINEL, errors.getvalue())

    def test_invalid_public_team_source_values_are_rejected(self):
        for value in ("raw", "blind", "random"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.config().configure(_base_argv("--team-source", value))


class TestBlindCanonicalStartup(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        self.state_directory = self.fixture.private_root / "state"
        self.state_directory.mkdir()
        self.state_path = self.state_directory / "bag.json"
        self.environ = {
            "TUGS_BLIND_POOL_ROOT": str(self.fixture.private_root),
            CANONICAL_REGISTRY_PATH_ENV: str(self.fixture.registry_path),
            "TUGS_BLIND_POOL_STATE": str(self.state_path),
        }
        self.config = load_blind_canonical_startup_config(self.environ)
        self.deployments = []

    def tearDown(self):
        for deployment in reversed(self.deployments):
            if deployment.owner_held:
                deployment.close()

    def prepare(self, **options):
        deployment = prepare_blind_canonical_deployment(
            self.config,
            owner_timeout_seconds=0.05,
            **options,
        )
        self.deployments.append(deployment)
        return deployment

    def assert_safe(self, error):
        rendered = str(error) + repr(error)
        for sentinel in (
            PRIVATE_SENTINEL,
            str(self.fixture.private_root),
            str(self.fixture.registry_path),
            str(self.state_path),
            TOKEN,
        ):
            self.assertNotIn(sentinel, rendered)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def test_required_environment_is_exact_and_raw_registry_is_not_substituted(self):
        raw_only = dict(self.environ)
        del raw_only[CANONICAL_REGISTRY_PATH_ENV]
        raw_only["TUGS_BLIND_POOL_REGISTRY"] = str(self.fixture.registry_path)
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            load_blind_canonical_startup_config(raw_only)
        self.assertEqual(
            BlindCanonicalErrorCategory.CONFIGURATION,
            caught.exception.category,
        )
        self.assert_safe(caught.exception)

    def test_missing_empty_and_relative_inputs_fail_safely(self):
        names = tuple(self.environ)
        for name in names:
            for value in (None, "", "relative-path"):
                values = dict(self.environ)
                if value is None:
                    del values[name]
                else:
                    values[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises(BlindCanonicalActivationError) as caught:
                        load_blind_canonical_startup_config(values)
                    self.assertEqual(
                        BlindCanonicalErrorCategory.CONFIGURATION,
                        caught.exception.category,
                    )
                    self.assert_safe(caught.exception)

    def test_first_start_creates_schema_two_under_owner_without_artifact_load(self):
        import fp.data.blind_pool.canonical_registry as registry_module

        real_read = registry_module._stable_read_bytes
        with mock.patch(
            "fp.data.blind_pool.canonical_artifacts.load_canonical_team_artifact",
            side_effect=AssertionError("full artifact read"),
        ), mock.patch.object(
            registry_module,
            "_stable_read_bytes",
            wraps=real_read,
        ) as stable_read:
            deployment = self.prepare()
        startup_reads = tuple(
            Path(call.args[0]).name for call in stable_read.call_args_list
        )
        self.assertNotIn("packed.txt", startup_reads)
        self.assertNotIn("team.json", startup_reads)
        self.assertIn("metadata.json", startup_reads)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(2, state["schema_version"])
        self.assertIsNone(state["reservation"])
        self.assertTrue(deployment.owner_held)
        self.assertEqual(2, deployment.active_count)
        deployment._store.snapshot()

    def test_owner_is_held_before_state_initialization(self):
        import fp.data.blind_pool.startup as startup_module

        events = []
        held_owner = None
        real_acquire = startup_module.acquire_blind_pool_deployment_owner
        real_initialize = BlindPoolBagStore.initialize_or_load

        def acquire(*args, **kwargs):
            nonlocal held_owner
            held_owner = real_acquire(*args, **kwargs)
            events.append("owner")
            return held_owner

        def initialize(store):
            self.assertIsNotNone(held_owner)
            self.assertTrue(held_owner.held)
            events.append("state")
            return real_initialize(store)

        with mock.patch.object(
            startup_module,
            "acquire_blind_pool_deployment_owner",
            side_effect=acquire,
        ), mock.patch.object(
            BlindPoolBagStore,
            "initialize_or_load",
            autospec=True,
            side_effect=initialize,
        ):
            self.prepare()
        self.assertEqual(["owner", "state"], events)

    def test_existing_idle_state_is_not_rewritten(self):
        first = self.prepare()
        first.close()
        before = self.state_path.read_bytes()
        second = self.prepare()
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertTrue(second.owner_held)

    def test_reserved_state_recovers_and_preserves_next_selection(self):
        first = self.prepare()
        reservation = first._store.reserve_next(BlindChallengeToken(TOKEN))
        first.close()
        second = self.prepare()
        self.assertIsNone(second._store.snapshot().reservation)
        again = second._store.reserve_next(BlindChallengeToken("b" * 32))
        self.assertEqual(reservation.team_id, again.team_id)
        second._store.release_reservation(again.reservation_id)

    def test_accept_sent_blocks_preparation_unchanged_and_releases_owner(self):
        first = self.prepare()
        reservation = first._store.reserve_next(BlindChallengeToken(TOKEN))
        first._store.mark_accept_sent(
            reservation.reservation_id, BlindChallengeToken(TOKEN)
        )
        first.close()
        before = self.state_path.read_bytes()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(
                self.config,
                owner_timeout_seconds=0.05,
            )
        self.assertEqual(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            caught.exception.category,
        )
        self.assertEqual(before, self.state_path.read_bytes())
        self.assert_safe(caught.exception)
        owner = acquire_blind_pool_deployment_owner(
            first._state_config,
            timeout_seconds=0.05,
        )
        owner.close()

    def test_schema_one_malformed_and_fingerprint_mismatch_preserve_bytes(self):
        first = self.prepare()
        first.close()
        original = json.loads(self.state_path.read_text(encoding="utf-8"))
        variants = (
            (
                {**original, "schema_version": 1},
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_state_invalid",
            ),
            (
                {**original, "registry_fingerprint": "f" * 64},
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                "blind_canonical_state_registry_mismatch",
            ),
            (
                None,
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_state_invalid",
            ),
        )
        for index, (document, category, code) in enumerate(variants):
            if document is None:
                payload = ("{" + PRIVATE_SENTINEL).encode()
            else:
                payload = (json.dumps(document, separators=(",", ":")) + "\n").encode()
            self.state_path.write_bytes(payload)
            with self.subTest(index=index):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    prepare_blind_canonical_deployment(
                        self.config,
                        owner_timeout_seconds=0.05,
                    )
                self.assertEqual(payload, self.state_path.read_bytes())
                self.assertEqual(category, caught.exception.category)
                self.assertEqual(code, caught.exception.code)
                self.assert_safe(caught.exception)

    def test_minimum_active_count_fails_before_state_or_owner_file(self):
        fixture = SyntheticCanonicalFixture()
        self.addCleanup(fixture.close)
        fixture.add_artifact("BL-101-v1")
        state_dir = fixture.private_root / "state"
        state_dir.mkdir()
        state = state_dir / "bag.json"
        config = BlindCanonicalStartupConfig(
            fixture.private_root,
            fixture.registry_path,
            state,
        )
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(config)
        self.assertEqual(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            caught.exception.category,
        )
        self.assertFalse(state.exists())
        self.assertFalse(state.with_name(state.name + ".owner.lock").exists())

    def test_owner_contention_blocks_then_releases_without_replacing_state_lock(self):
        first = self.prepare()
        first._store.snapshot()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(
                self.config,
                owner_timeout_seconds=0.05,
            )
        self.assertEqual(
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            caught.exception.category,
        )
        first.close()
        second = self.prepare()
        self.assertTrue(second.owner_held)

    def test_stale_regular_owner_control_does_not_block(self):
        owner_path = self.state_path.with_name(self.state_path.name + ".owner.lock")
        owner_path.touch()
        deployment = self.prepare()
        self.assertTrue(deployment.owner_held)

    def test_owner_rejects_hardlinked_control_file(self):
        owner_path = self.state_path.with_name(self.state_path.name + ".owner.lock")
        os.link(self.fixture.registry_path, owner_path)
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.fixture.private_root, self.fixture.registry_path),
            self.state_path,
        )
        with self.assertRaises(BlindPoolValidationError) as caught:
            acquire_blind_pool_deployment_owner(state_config, timeout_seconds=0.05)
        self.assertEqual("deployment_owner_path_invalid", caught.exception.code)

    def test_owner_descriptor_identity_failure_releases_lock(self):
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.fixture.private_root, self.fixture.registry_path),
            self.state_path,
        )
        with mock.patch(
            "fp.data.blind_pool.ownership._same_file_object",
            return_value=False,
        ):
            with self.assertRaises(BlindPoolValidationError) as caught:
                acquire_blind_pool_deployment_owner(
                    state_config,
                    timeout_seconds=0.05,
                )
        self.assertEqual("deployment_ownership_unavailable", caught.exception.code)
        owner = acquire_blind_pool_deployment_owner(
            state_config,
            timeout_seconds=0.05,
        )
        owner.close()

    def test_owner_configuration_failure_has_no_private_exception_context(self):
        missing_registry = self.fixture.private_root / PRIVATE_SENTINEL
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.fixture.private_root, missing_registry),
            self.state_path,
        )
        with self.assertRaises(BlindPoolValidationError) as caught:
            acquire_blind_pool_deployment_owner(state_config, timeout_seconds=0.05)
        self.assertEqual("deployment_owner_config_invalid", caught.exception.code)
        self.assertNotIn(PRIVATE_SENTINEL, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_owner_rejects_link_or_reparse_control_where_supported(self):
        owner_path = self.state_path.with_name(self.state_path.name + ".owner.lock")
        target = self.state_directory / "synthetic-target.lock"
        target.touch()
        try:
            owner_path.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest("file symlink unavailable: {}".format(type(error).__name__))
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.fixture.private_root, self.fixture.registry_path),
            self.state_path,
        )
        with self.assertRaises(BlindPoolValidationError) as caught:
            acquire_blind_pool_deployment_owner(state_config, timeout_seconds=0.05)
        self.assertEqual("deployment_owner_path_invalid", caught.exception.code)

    @unittest.skipUnless(os.name == "nt", "Windows owner-release proof")
    def test_owner_os_lock_releases_after_process_termination(self):
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(self.fixture.private_root, self.fixture.registry_path),
            self.state_path,
        )
        script = (
            "import sys,time;from pathlib import Path;"
            "from fp.data.blind_pool.models import BlindPoolConfig,BlindPoolStateConfig;"
            "from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner;"
            "c=BlindPoolStateConfig(BlindPoolConfig(Path(sys.argv[1]),Path(sys.argv[2])),Path(sys.argv[3]));"
            "g=acquire_blind_pool_deployment_owner(c,timeout_seconds=1);"
            "print('ready',flush=True);time.sleep(60)"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                str(self.fixture.private_root),
                str(self.fixture.registry_path),
                str(self.state_path),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            self.assertEqual("ready", process.stdout.readline().strip())
            with self.assertRaises(BlindPoolValidationError):
                acquire_blind_pool_deployment_owner(
                    state_config,
                    timeout_seconds=0.05,
                )
        finally:
            process.terminate()
            process.wait(timeout=5)
        owner = acquire_blind_pool_deployment_owner(
            state_config,
            timeout_seconds=0.5,
        )
        owner.close()

    def test_startup_objects_are_immutable_safe_and_nonserializable(self):
        deployment = self.prepare()
        owner = deployment._owner
        rendered = "".join(
            repr(value) + str(value) for value in (self.config, deployment, owner)
        )
        for forbidden in (
            str(self.fixture.private_root),
            str(self.fixture.registry_path),
            str(self.state_path),
            deployment._registry.registry_fingerprint,
            "BL-001-v1",
        ):
            self.assertNotIn(forbidden, rendered)
        for value in (self.config, deployment, owner):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    pickle.dumps(value)


class _FakeRuntime:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def run_once(self):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakePrepared:
    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = False
        self.create_calls = []

    def create_runtime(self, *args):
        self.create_calls.append(args)
        return self.runtime

    def close(self):
        self.closed = True


class _FakeClient:
    username = "SyntheticBot"

    def __init__(self, *, login_error=None):
        self.login_error = login_error
        self.closed = False
        self.controls = []

    async def login(self):
        if self.login_error is not None:
            raise self.login_error
        return self.username

    async def avatar(self, _avatar):
        return None

    async def update_team(self, _packed):
        return None

    async def close(self):
        self.closed = True


class _ExactSyntheticClient(FakeTransport):
    def __init__(self, messages):
        super().__init__(messages)
        self.closed = False
        self.submitted = []

    async def login(self):
        return self.username

    async def avatar(self, _avatar):
        return None

    async def update_team(self, packed):
        self.submitted.append(packed)

    async def close(self):
        self.closed = True


def _process_config(run_count=2):
    return SimpleNamespace(
        username="SyntheticBot",
        password=None,
        websocket_uri="ws://synthetic.invalid",
        local_no_security_login=False,
        avatar=None,
        pokemon_format="gen9tugs",
        run_count=run_count,
        user_id=None,
    )


class TestBlindCanonicalActivation(unittest.IsolatedAsyncioTestCase):
    async def test_local_preflight_precedes_connect_and_two_battles_reuse_runtime(self):
        events = []
        runtime = _FakeRuntime(("SyntheticBot", "Other"))
        prepared = _FakePrepared(runtime)
        client = _FakeClient()

        def load_config(_environ):
            events.append("config")
            return mock.sentinel.startup_config

        def prepare(_config):
            events.append("prepare")
            return prepared

        async def connect(*_args):
            events.append("connect")
            return client

        checks = []
        with mock.patch(
            "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
            side_effect=load_config,
        ), mock.patch(
            "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
            side_effect=prepare,
        ):
            await run_blind_canonical_activation(
                _process_config(),
                mock.sentinel.public_prior,
                mock.sentinel.pokedex,
                mock.sentinel.moves,
                websocket_factory=connect,
                environ={},
                battle_runner=mock.AsyncMock(),
                integrity_checker=lambda *args: checks.append(args),
            )
        self.assertEqual(["config", "prepare", "connect"], events)
        self.assertEqual(2, runtime.calls)
        self.assertEqual(1, len(prepared.create_calls))
        self.assertEqual(2, len(checks))
        self.assertTrue(client.closed)
        self.assertTrue(prepared.closed)
        self.assertEqual([], client.controls)

    async def test_every_preparation_failure_is_before_websocket_construction(self):
        categories = (
            BlindCanonicalErrorCategory.CONFIGURATION,
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
        )
        for category in categories:
            connect = mock.AsyncMock()
            error = BlindCanonicalActivationError(category, "safe_code", "Safe failure")
            with self.subTest(category=category.value), mock.patch(
                "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
                return_value=mock.sentinel.config,
            ), mock.patch(
                "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
                side_effect=error,
            ):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    await run_blind_canonical_activation(
                        _process_config(),
                        None,
                        None,
                        None,
                        websocket_factory=connect,
                        environ={},
                        battle_runner=mock.AsyncMock(),
                        integrity_checker=mock.Mock(),
                    )
            self.assertEqual(category, caught.exception.category)
            connect.assert_not_awaited()

    async def test_login_failure_is_transient_idle_and_releases_owner(self):
        prepared = _FakePrepared(_FakeRuntime(()))
        client = _FakeClient(login_error=OSError(PRIVATE_SENTINEL))
        with mock.patch(
            "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
            return_value=mock.sentinel.config,
        ), mock.patch(
            "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
            return_value=prepared,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                await run_blind_canonical_activation(
                    _process_config(),
                    None,
                    None,
                    None,
                    websocket_factory=mock.AsyncMock(return_value=client),
                    environ={},
                    battle_runner=mock.AsyncMock(),
                    integrity_checker=mock.Mock(),
                )
        self.assertEqual(
            BlindCanonicalErrorCategory.TRANSIENT_NETWORK,
            caught.exception.category,
        )
        self.assertNotIn(PRIVATE_SENTINEL, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(client.closed)
        self.assertTrue(prepared.closed)
        self.assertEqual(0, prepared.runtime.calls)

    async def test_programmer_error_remains_distinct_and_releases_owner(self):
        prepared = _FakePrepared(_FakeRuntime(()))
        connect = mock.AsyncMock(side_effect=AttributeError(PRIVATE_SENTINEL))
        with mock.patch(
            "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
            return_value=mock.sentinel.config,
        ), mock.patch(
            "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
            return_value=prepared,
        ):
            with self.assertRaisesRegex(AttributeError, PRIVATE_SENTINEL):
                await run_blind_canonical_activation(
                    _process_config(),
                    None,
                    None,
                    None,
                    websocket_factory=connect,
                    environ={},
                    battle_runner=mock.AsyncMock(),
                    integrity_checker=mock.Mock(),
                )
        self.assertTrue(prepared.closed)

    async def test_cancellation_during_socket_close_still_releases_owner(self):
        prepared = _FakePrepared(_FakeRuntime((asyncio.CancelledError(),)))
        client = _FakeClient()

        async def cancelled_close():
            raise asyncio.CancelledError

        client.close = cancelled_close
        with mock.patch(
            "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
            return_value=mock.sentinel.config,
        ), mock.patch(
            "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
            return_value=prepared,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_blind_canonical_activation(
                    _process_config(),
                    None,
                    None,
                    None,
                    websocket_factory=mock.AsyncMock(return_value=client),
                    environ={},
                    battle_runner=mock.AsyncMock(),
                    integrity_checker=mock.Mock(),
                )
        self.assertTrue(prepared.closed)
        self.assertEqual(1, prepared.runtime.calls)

    async def test_reconciliation_and_preaccept_failures_halt_without_next_attempt(
        self,
    ):
        errors = (
            (
                BlindPoolReconciliationRequired("reconciliation_required", "safe"),
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            ),
            (
                BlindPoolLifecycleError(
                    "team_artifact_verification_failed",
                    "safe",
                ),
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            ),
            (
                BlindPoolLifecycleError("team_submission_failed", "safe"),
                BlindCanonicalErrorCategory.TRANSIENT_NETWORK,
            ),
        )
        for failure, category in errors:
            runtime = _FakeRuntime((failure, "SyntheticBot"))
            prepared = _FakePrepared(runtime)
            client = _FakeClient()
            with self.subTest(code=failure.code), mock.patch(
                "fp.data.blind_pool.activation.load_blind_canonical_startup_config",
                return_value=mock.sentinel.config,
            ), mock.patch(
                "fp.data.blind_pool.activation.prepare_blind_canonical_deployment",
                return_value=prepared,
            ):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    await run_blind_canonical_activation(
                        _process_config(),
                        None,
                        None,
                        None,
                        websocket_factory=mock.AsyncMock(return_value=client),
                        environ={},
                        battle_runner=mock.AsyncMock(),
                        integrity_checker=mock.Mock(),
                    )
            self.assertEqual(category, caught.exception.category)
            self.assertEqual(1, runtime.calls)
            self.assertTrue(prepared.closed)
            self.assertTrue(client.closed)


class TestBlindCanonicalMainOrchestration(unittest.TestCase):
    def test_main_canonical_branch_uses_common_initialization_then_activation(self):
        import fp.main as main_module

        values = {
            "team_source": TEAM_SOURCE_BLIND_CANONICAL,
            "log_level": "INFO",
            "log_to_file": False,
            "pokemon_format": "gen9tugs",
        }
        events = []
        with mock.patch.multiple(
            FoulPlayConfig, create=True, **values
        ), mock.patch.object(
            FoulPlayConfig,
            "configure",
            return_value=mock.sentinel.options,
        ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
            main_module,
            "apply_mods",
            side_effect=lambda _spec: events.append("mods"),
        ), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            side_effect=lambda *_args: events.append("prior") or mock.sentinel.prior,
        ), mock.patch.object(
            main_module,
            "run_blind_canonical_activation",
            new=mock.AsyncMock(
                side_effect=lambda *_args, **_kwargs: events.append("blind")
            ),
        ) as activation:
            asyncio.run(main_module.run_foul_play(environ={}))
        self.assertEqual(["mods", "prior", "blind"], events)
        activation.assert_awaited_once()

    def test_legacy_main_ignores_malformed_blind_environment(self):
        import fp.main as main_module

        client = mock.AsyncMock()
        client.login.return_value = "SyntheticBot"
        values = {
            "team_source": TEAM_SOURCE_LEGACY,
            "log_level": "INFO",
            "log_to_file": False,
            "username": "SyntheticBot",
            "password": None,
            "websocket_uri": "ws://synthetic.invalid",
            "local_no_security_login": False,
            "avatar": None,
            "team_list": None,
            "bot_mode": BotModes.search_ladder,
            "pokemon_format": "gen9tugs",
            "run_count": 1,
        }
        mode = SimpleNamespace(requires_team=False)
        invalid_environment = {
            "TUGS_BLIND_POOL_ROOT": PRIVATE_SENTINEL,
            "TUGS_BLIND_POOL_REGISTRY": PRIVATE_SENTINEL,
            "TUGS_BLIND_POOL_STATE": PRIVATE_SENTINEL,
            CANONICAL_REGISTRY_PATH_ENV: PRIVATE_SENTINEL,
        }
        with mock.patch.multiple(
            FoulPlayConfig, create=True, **values
        ), mock.patch.object(
            FoulPlayConfig,
            "configure",
            return_value=None,
        ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
            main_module, "apply_mods"
        ), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            return_value=None,
        ), mock.patch.object(
            main_module, "battle_mode", return_value=mode
        ), mock.patch.object(
            main_module,
            "pokemon_battle",
            new=mock.AsyncMock(return_value="SyntheticBot"),
        ), mock.patch.object(
            main_module,
            "run_blind_canonical_activation",
            new=mock.AsyncMock(),
        ) as activation:
            asyncio.run(
                main_module.run_foul_play(
                    websocket_factory=mock.AsyncMock(return_value=client),
                    environ=invalid_environment,
                )
            )
        activation.assert_not_awaited()
        client.update_team.assert_awaited_once_with("None")
        client.search_for_match.assert_awaited_once_with("gen9tugs")

    def test_expected_canonical_failure_exits_without_private_traceback(self):
        import fp.main as main_module

        error = BlindCanonicalActivationError(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_canonical_recovery_required",
            "Safe recovery required",
        )
        values = {
            "team_source": TEAM_SOURCE_BLIND_CANONICAL,
            "log_level": "INFO",
            "log_to_file": False,
            "pokemon_format": "gen9tugs",
        }
        with mock.patch.multiple(
            FoulPlayConfig, create=True, **values
        ), mock.patch.object(
            FoulPlayConfig, "configure", return_value=None
        ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
            main_module, "apply_mods"
        ), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            return_value=None,
        ), mock.patch.object(
            main_module,
            "run_blind_canonical_activation",
            new=mock.AsyncMock(side_effect=error),
        ), self.assertLogs("fp.main", logging.ERROR) as logs:
            with self.assertRaises(SystemExit) as caught:
                asyncio.run(main_module.run_foul_play(environ={}))
        self.assertEqual(1, caught.exception.code)
        rendered = "\n".join(logs.output)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertNotIn("Traceback", rendered)

    def test_imports_have_no_canonical_side_effects(self):
        invalid = {
            "TUGS_BLIND_POOL_ROOT": PRIVATE_SENTINEL,
            "TUGS_BLIND_POOL_STATE": PRIVATE_SENTINEL,
            CANONICAL_REGISTRY_PATH_ENV: PRIVATE_SENTINEL,
        }
        with mock.patch.dict(os.environ, invalid, clear=False), mock.patch(
            "pathlib.Path.open",
            side_effect=AssertionError("filesystem touched"),
        ):
            for module_name in (
                "fp.config",
                "fp.main",
                "fp.data.blind_pool.startup",
                "fp.data.blind_pool.activation",
                "fp.data.blind_pool.ownership",
                "fp.data.blind_pool",
            ):
                importlib.import_module(module_name)


class TestBlindCanonicalTopLevel(unittest.TestCase):
    setUp = TestBlindCanonicalStartup.setUp
    tearDown = TestBlindCanonicalStartup.tearDown

    def test_actual_main_branch_preflights_then_runs_one_exact_synthetic_battle(self):
        import fp.main as main_module

        client = _ExactSyntheticClient((EXACT_CHALLENGE, exact_room_message()))
        events = []
        values = {
            "team_source": TEAM_SOURCE_BLIND_CANONICAL,
            "log_level": "INFO",
            "log_to_file": False,
            "username": "Blind Bot",
            "password": None,
            "websocket_uri": "ws://synthetic.invalid",
            "local_no_security_login": False,
            "avatar": None,
            "pokemon_format": "gen9tugs",
            "run_count": 1,
        }

        async def connect(*_args):
            events.append("connect")
            self.assertEqual(0, artifact_loader.call_count)
            self.assertTrue(self.state_path.is_file())
            self.assertIsNone(json.loads(self.state_path.read_text())["reservation"])
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                prepare_blind_canonical_deployment(
                    self.config,
                    owner_timeout_seconds=0.05,
                )
            self.assertEqual(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                caught.exception.category,
            )
            return client

        async def battle_runner(_client, _format_id, projection, **kwargs):
            events.append("battle")
            self.assertEqual(6, len(projection))
            self.assertIs(mock.sentinel.prior, kwargs["public_prior_configuration"])
            return "Blind Bot"

        import fp.data.blind_pool.canonical_runtime as runtime_module

        real_loader = runtime_module.load_canonical_team_artifact
        with mock.patch.multiple(
            FoulPlayConfig, create=True, **values
        ), mock.patch.object(
            FoulPlayConfig,
            "configure",
            return_value=mock.sentinel.options,
        ), mock.patch.object(main_module, "init_logging"), mock.patch.object(
            main_module, "apply_mods", side_effect=lambda _spec: events.append("mods")
        ), mock.patch.object(
            main_module,
            "load_public_prior_runtime_configuration",
            side_effect=lambda *_args: events.append("prior") or mock.sentinel.prior,
        ), mock.patch.object(
            main_module,
            "pokemon_battle",
            new=mock.AsyncMock(side_effect=battle_runner),
        ), mock.patch.object(
            runtime_module,
            "load_canonical_team_artifact",
            wraps=real_loader,
        ) as artifact_loader:
            asyncio.run(
                main_module.run_foul_play(
                    websocket_factory=connect,
                    environ=self.environ,
                )
            )

        self.assertEqual(["mods", "prior", "connect", "battle"], events)
        self.assertEqual(1, artifact_loader.call_count)
        self.assertEqual(["enable"], client.controls)
        self.assertEqual(1, len(client.sent))
        self.assertEqual(1, len(client.submitted))
        self.assertTrue(client.closed)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertIsNone(state["reservation"])
        self.assertEqual(1, state["next_index"])


class TestBlindCanonicalLogPrivacy(unittest.TestCase):
    setUp = TestBlindCanonicalStartup.setUp
    tearDown = TestBlindCanonicalStartup.tearDown
    prepare = TestBlindCanonicalStartup.prepare

    def test_normal_preparation_and_bag_logs_hide_opaque_ids_and_private_values(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        package_logger = logging.getLogger("fp.data.blind_pool")
        previous_level = package_logger.level
        package_logger.setLevel(logging.INFO)
        package_logger.addHandler(handler)
        try:
            deployment = self.prepare()
            reservation = deployment._store.reserve_next(BlindChallengeToken(TOKEN))
            deployment._store.release_reservation(reservation.reservation_id)
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(previous_level)
        rendered = output.getvalue()
        for forbidden in (
            "BL-001-v1",
            "BL-002-v1",
            TOKEN,
            str(self.fixture.private_root),
            str(self.fixture.registry_path),
            str(self.state_path),
            deployment._registry.registry_fingerprint,
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("active pool count: 2", rendered)
