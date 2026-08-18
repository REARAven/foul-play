from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import requests

import fp.data.blind_pool.bag as bag_module
import fp.data.blind_pool.canonical_registry as canonical_registry
import fp.data.blind_pool.maintenance as maintenance
from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.maintenance import (
    BlindCanonicalInspectionStatus,
    inspect_blind_canonical_deployment,
    reconcile_blind_canonical_deployment,
)
from fp.data.blind_pool.models import BlindChallengeToken
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.reconciliation import (
    RECONCILIATION_CASE_DOMAIN,
    BlindReconciliationDisposition,
    derive_reconciliation_case,
)
from fp.data.blind_pool.rating_state import (
    BlindRatingStateStore,
    validate_rating_state_config,
)
from fp.data.blind_pool.result_ledger import (
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from fp.data.blind_pool.startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    prepare_blind_canonical_deployment,
)
from fp.websocket_client import PSWebsocketClient
from tests.test_blind_canonical_maintenance import MaintenanceFixture
from tests.test_blind_pool_bag import BlindPoolBagFixture, FailingTextStream


ROOT = Path(__file__).resolve().parents[1]
TOKEN_A = "a" * 32
TOKEN_B = "b" * 32


class ReconciliationFixture(MaintenanceFixture):
    def setUp(self):
        super().setUp()
        result_directory = self.fixture.base / "results"
        result_directory.mkdir()
        self.result_path = result_directory / "ledger.json"
        self.rating_path = result_directory / "ratings.json"
        self.config = replace(
            self.config,
            result_ledger_path=self.result_path,
            rating_state_path=self.rating_path,
        )
        result_store = BlindResultLedgerStore(
            validate_result_ledger_config(
                self.result_path,
                private_root=self.fixture.private_root,
                registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
            )
        )
        result_store.initialize_empty()
        rating_store = BlindRatingStateStore(
            validate_rating_state_config(
                self.rating_path,
                private_root=self.fixture.private_root,
                registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
                result_ledger_path=self.result_path,
            )
        )
        rating_store.initialize(result_store.require_ready())

    def quarantine(self, token: str = TOKEN_A):
        store = self.initialize()
        reservation = store.reserve_next(BlindChallengeToken(token))
        store.mark_accept_sent(
            reservation.reservation_id,
            BlindChallengeToken(token),
        )
        return store, reservation

    def current_case(self):
        result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.RECOVERY_REQUIRED, result.status
        )
        self.assertIsNotNone(result.reconciliation_case)
        return result.reconciliation_case

    def resolve_args(self, command, case):
        disposition = "consumed" if command == "resolve-consumed" else "not-consumed"
        return self.cli_args(command) + [
            "--case",
            case,
            "--confirm",
            disposition,
        ]


class TestReconciliationCase(ReconciliationFixture):
    def test_case_is_deterministic_domain_separated_and_private_identity_bound(self):
        store, _reservation = self.quarantine()
        state = store.snapshot()
        first = derive_reconciliation_case(state)
        self.assertEqual(first, derive_reconciliation_case(state))
        self.assertEqual(first, derive_reconciliation_case(store.snapshot()))
        self.assertEqual(first, self.current_case())
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(b"TUGS-BLIND-RECONCILIATION-v1", RECONCILIATION_CASE_DOMAIN)
        self.assertNotEqual(first, TOKEN_A)
        self.assertNotIn(TOKEN_A, first)

        changed_token = replace(
            state,
            reservation=replace(
                state.reservation,
                challenge_token=BlindChallengeToken(TOKEN_B),
            ),
        )
        changed_team = replace(
            state,
            reservation=replace(state.reservation, team_id="BL-999-v1"),
        )
        changed_reservation = replace(
            state,
            reservation=replace(state.reservation, reservation_id="f" * 32),
        )
        changed_fingerprint = replace(state, registry_fingerprint="f" * 64)
        for changed in (
            changed_token,
            changed_team,
            changed_reservation,
            changed_fingerprint,
        ):
            self.assertNotEqual(first, derive_reconciliation_case(changed))
        with self.assertRaises(BlindPoolValidationError) as caught:
            derive_reconciliation_case(
                replace(
                    state,
                    reservation=replace(state.reservation, phase="reserved"),
                )
            )
        self.assertEqual("reconciliation_not_applicable", caught.exception.code)

    def test_status_matrix_is_read_only_and_only_accept_sent_has_case(self):
        absent = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.READY_FOR_FIRST_START,
            absent.status,
        )
        self.assertIsNone(absent.reconciliation_case)

        store = self.initialize()
        idle_bytes = self.state_path.read_bytes()
        idle = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(BlindCanonicalInspectionStatus.READY, idle.status)
        self.assertIsNone(idle.reconciliation_case)
        self.assertEqual(idle_bytes, self.state_path.read_bytes())

        reservation = store.reserve_next(BlindChallengeToken(TOKEN_A))
        reserved_bytes = self.state_path.read_bytes()
        reserved = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.STARTUP_RECOVERY_AVAILABLE,
            reserved.status,
        )
        self.assertIsNone(reserved.reconciliation_case)
        self.assertEqual(reserved_bytes, self.state_path.read_bytes())

        store.mark_accept_sent(reservation.reservation_id, BlindChallengeToken(TOKEN_A))
        quarantined_bytes = self.state_path.read_bytes()
        quarantined = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.RECOVERY_REQUIRED,
            quarantined.status,
        )
        self.assertRegex(quarantined.reconciliation_case, r"^[0-9a-f]{64}$")
        self.assertEqual(quarantined_bytes, self.state_path.read_bytes())

    def test_status_cli_prints_case_but_never_token_or_team(self):
        self.quarantine()
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = maintenance.main(self.cli_args("status"))
        rendered = output.getvalue() + errors.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("deployment status: recovery_required", rendered)
        self.assertRegex(rendered, r"reconciliation case: [0-9a-f]{64}")
        self.assertNotIn(TOKEN_A, rendered)
        self.assertNotIn("BL-", rendered)

    def test_missing_accept_sent_token_fails_without_misleading_case(self):
        self.quarantine()
        document = json.loads(self.state_path.read_text(encoding="utf-8"))
        document["reservation"]["challenge_token"] = None
        payload = (json.dumps(document, separators=(",", ":")) + "\n").encode()
        self.state_path.write_bytes(payload)
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            caught.exception.category,
        )
        self.assertNotIn(TOKEN_A, str(caught.exception) + repr(caught.exception))
        self.assertEqual(payload, self.state_path.read_bytes())


class TestOfflineReconciliationTransitions(ReconciliationFixture):
    def test_consumed_uses_normal_advancement_exactly_once(self):
        store, reservation = self.quarantine()
        before = store.snapshot()
        case = derive_reconciliation_case(before)
        result = reconcile_blind_canonical_deployment(
            self.config,
            case,
            BlindReconciliationDisposition.CONSUMED,
        )
        self.assertEqual(BlindReconciliationDisposition.CONSUMED, result.disposition)
        after = store.snapshot()
        self.assertIsNone(after.reservation)
        self.assertEqual(before.next_index + 1, after.next_index)
        self.assertEqual(reservation.team_id, after.last_consumed_id)
        self.assertEqual(
            BlindCanonicalInspectionStatus.READY,
            inspect_blind_canonical_deployment(self.config).status,
        )
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            reconcile_blind_canonical_deployment(
                self.config,
                case,
                BlindReconciliationDisposition.CONSUMED,
            )
        self.assertEqual(
            BlindCanonicalErrorCategory.RECONCILIATION_NOT_APPLICABLE,
            caught.exception.category,
        )
        self.assertEqual(after, store.snapshot())
        next_reservation = store.reserve_next(BlindChallengeToken(TOKEN_B))
        self.assertEqual(
            before.cycle_order[before.next_index + 1], next_reservation.team_id
        )

    def test_not_consumed_releases_same_position_without_advancement(self):
        store, reservation = self.quarantine()
        before = store.snapshot()
        case = derive_reconciliation_case(before)
        result = reconcile_blind_canonical_deployment(
            self.config,
            case,
            BlindReconciliationDisposition.NOT_CONSUMED,
        )
        self.assertEqual(
            BlindReconciliationDisposition.NOT_CONSUMED,
            result.disposition,
        )
        after = store.snapshot()
        self.assertIsNone(after.reservation)
        self.assertEqual(before.next_index, after.next_index)
        self.assertEqual(before.last_consumed_id, after.last_consumed_id)
        retry = store.reserve_next(BlindChallengeToken(TOKEN_B))
        self.assertEqual(reservation.team_id, retry.team_id)

    def test_runtime_release_still_rejects_accept_sent(self):
        store, reservation = self.quarantine()
        before = self.state_path.read_bytes()
        with self.assertRaises(BlindPoolValidationError) as caught:
            store.release_reservation(reservation.reservation_id)
        self.assertEqual("reservation_phase_transition_invalid", caught.exception.code)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_exact_case_grammar_and_wrong_case_fail_without_mutation(self):
        self.quarantine()
        valid_case = self.current_case()
        invalid_cases = (
            valid_case[:-1],
            valid_case + "0",
            valid_case.upper(),
            "g" * 64,
            " " + valid_case,
            valid_case + " ",
            "sha256:" + valid_case,
            "",
        )
        for invalid in invalid_cases:
            before = self.state_path.read_bytes()
            with self.subTest(case=invalid[:8]):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    reconcile_blind_canonical_deployment(
                        self.config,
                        invalid,
                        BlindReconciliationDisposition.CONSUMED,
                    )
                self.assertEqual(
                    BlindCanonicalErrorCategory.RECONCILIATION_MISMATCH,
                    caught.exception.category,
                )
                self.assertEqual(before, self.state_path.read_bytes())
        wrong = "0" * 64 if valid_case != "0" * 64 else "1" * 64
        before = self.state_path.read_bytes()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            reconcile_blind_canonical_deployment(
                self.config,
                wrong,
                BlindReconciliationDisposition.NOT_CONSUMED,
            )
        self.assertEqual(
            BlindCanonicalErrorCategory.RECONCILIATION_MISMATCH,
            caught.exception.category,
        )
        self.assertEqual(before, self.state_path.read_bytes())

    def test_absent_idle_and_reserved_are_not_applicable(self):
        for phase in ("absent", "idle", "reserved"):
            with self.subTest(phase=phase):
                if phase == "idle":
                    self.initialize()
                elif phase == "reserved":
                    self.initialize().reserve_next(BlindChallengeToken(TOKEN_A))
                before = (
                    self.state_path.read_bytes() if self.state_path.exists() else None
                )
                for disposition in BlindReconciliationDisposition:
                    with self.assertRaises(BlindCanonicalActivationError) as caught:
                        reconcile_blind_canonical_deployment(
                            self.config,
                            "0" * 64,
                            disposition,
                        )
                    self.assertEqual(
                        BlindCanonicalErrorCategory.RECONCILIATION_NOT_APPLICABLE,
                        caught.exception.category,
                    )
                    after = (
                        self.state_path.read_bytes()
                        if self.state_path.exists()
                        else None
                    )
                    self.assertEqual(before, after)

    def test_schema_v1_and_fingerprint_mismatch_never_mutate(self):
        self.quarantine()
        valid = json.loads(self.state_path.read_text(encoding="utf-8"))
        for document, category in (
            (
                {**valid, "schema_version": 1},
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            ),
            (
                {**valid, "registry_fingerprint": "f" * 64},
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            ),
        ):
            payload = (json.dumps(document, separators=(",", ":")) + "\n").encode()
            self.state_path.write_bytes(payload)
            with self.subTest(category=category.value):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    reconcile_blind_canonical_deployment(
                        self.config,
                        "0" * 64,
                        BlindReconciliationDisposition.CONSUMED,
                    )
                self.assertEqual(category, caught.exception.category)
                self.assertEqual(payload, self.state_path.read_bytes())

    def test_cross_disposition_reuse_of_resolved_case_never_mutates(self):
        for first, second in (
            (
                BlindReconciliationDisposition.CONSUMED,
                BlindReconciliationDisposition.NOT_CONSUMED,
            ),
            (
                BlindReconciliationDisposition.NOT_CONSUMED,
                BlindReconciliationDisposition.CONSUMED,
            ),
        ):
            with self.subTest(first=first.value, second=second.value):
                if self.state_path.exists():
                    self.state_path.unlink()
                self.quarantine()
                case = self.current_case()
                reconcile_blind_canonical_deployment(self.config, case, first)
                before = self.state_path.read_bytes()
                with self.assertRaises(BlindCanonicalActivationError):
                    reconcile_blind_canonical_deployment(self.config, case, second)
                self.assertEqual(before, self.state_path.read_bytes())

    def test_stale_case_is_rechecked_under_transaction_lock(self):
        store, _reservation = self.quarantine()
        observed = store.snapshot()
        old_case = derive_reconciliation_case(observed)
        changed = replace(
            observed,
            reservation=replace(
                observed.reservation,
                challenge_token=BlindChallengeToken(TOKEN_B),
            ),
        )
        real_derive = maintenance.derive_reconciliation_case
        calls = 0

        def change_after_initial_read(state):
            nonlocal calls
            calls += 1
            case = real_derive(state)
            if calls == 1:
                bag_module.write_blind_pool_bag_state_atomic(
                    store._config,
                    changed,
                    store._selection,
                )
            return case

        with mock.patch.object(
            maintenance,
            "derive_reconciliation_case",
            side_effect=change_after_initial_read,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                reconcile_blind_canonical_deployment(
                    self.config,
                    old_case,
                    BlindReconciliationDisposition.CONSUMED,
                )
        self.assertEqual(
            BlindCanonicalErrorCategory.RECONCILIATION_MISMATCH,
            caught.exception.category,
        )
        self.assertEqual(changed, store.snapshot())

    def test_cycle_boundary_consumption_preserves_no_repeat_rule(self):
        store = self.initialize()
        first = store.reserve_next(BlindChallengeToken(TOKEN_A))
        store.commit_reservation(first.reservation_id)
        second = store.reserve_next(BlindChallengeToken(TOKEN_B))
        store.mark_accept_sent(second.reservation_id, BlindChallengeToken(TOKEN_B))
        case = derive_reconciliation_case(store.snapshot())
        reconcile_blind_canonical_deployment(
            self.config,
            case,
            BlindReconciliationDisposition.CONSUMED,
        )
        next_cycle = store.reserve_next(BlindChallengeToken(TOKEN_A))
        self.assertEqual(2, next_cycle.cycle_number)
        self.assertNotEqual(second.team_id, next_cycle.team_id)

    def test_cycle_end_not_consumed_does_not_roll_or_reshuffle(self):
        store = self.initialize()
        first = store.reserve_next(BlindChallengeToken(TOKEN_A))
        store.commit_reservation(first.reservation_id)
        second = store.reserve_next(BlindChallengeToken(TOKEN_B))
        store.mark_accept_sent(second.reservation_id, BlindChallengeToken(TOKEN_B))
        before = store.snapshot()
        reconcile_blind_canonical_deployment(
            self.config,
            derive_reconciliation_case(before),
            BlindReconciliationDisposition.NOT_CONSUMED,
        )
        after = store.snapshot()
        self.assertEqual(before.cycle_number, after.cycle_number)
        self.assertEqual(before.cycle_order, after.cycle_order)
        self.assertEqual(before.next_index, after.next_index)
        retry = store.reserve_next(BlindChallengeToken(TOKEN_A))
        self.assertEqual(second.team_id, retry.team_id)

    def test_atomic_failures_before_replace_preserve_state_and_release_owner(self):
        for disposition in BlindReconciliationDisposition:
            for failure_point in ("serialize", "write", "flush", "fsync", "replace"):
                with self.subTest(
                    disposition=disposition.value,
                    failure_point=failure_point,
                ):
                    if self.state_path.exists():
                        self.state_path.unlink()
                    self.quarantine()
                    case = self.current_case()
                    before = self.state_path.read_bytes()
                    original_fdopen = os.fdopen

                    def failing_fdopen(*args, **kwargs):
                        return FailingTextStream(
                            original_fdopen(*args, **kwargs),
                            failure_point,
                        )

                    if failure_point == "serialize":
                        patcher = mock.patch(
                            "fp.data.blind_pool.state._serialize_state",
                            side_effect=TypeError("synthetic serialization failure"),
                        )
                    elif failure_point in {"write", "flush"}:
                        patcher = mock.patch(
                            "fp.data.blind_pool.state.os.fdopen",
                            side_effect=failing_fdopen,
                        )
                    elif failure_point == "fsync":
                        patcher = mock.patch(
                            "fp.data.blind_pool.state.os.fsync",
                            side_effect=OSError("synthetic fsync failure"),
                        )
                    else:
                        patcher = mock.patch(
                            "fp.data.blind_pool.state.os.replace",
                            side_effect=OSError("synthetic replace failure"),
                        )
                    with patcher:
                        with self.assertRaises(BlindCanonicalActivationError) as caught:
                            reconcile_blind_canonical_deployment(
                                self.config,
                                case,
                                disposition,
                            )
                    self.assertEqual(
                        BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                        caught.exception.category,
                    )
                    self.assertEqual(before, self.state_path.read_bytes())
                    self.assertEqual(
                        [],
                        list(self.state_path.parent.glob(".*.tmp")),
                    )
                    owner = acquire_blind_pool_deployment_owner(
                        self.state_config(),
                        timeout_seconds=0.05,
                    )
                    owner.close()

    def test_post_replace_failure_leaves_only_complete_resolved_state(self):
        for disposition in BlindReconciliationDisposition:
            with self.subTest(disposition=disposition.value):
                if self.state_path.exists():
                    self.state_path.unlink()
                store, reservation = self.quarantine()
                before = store.snapshot()
                case = derive_reconciliation_case(before)
                with mock.patch(
                    "fp.data.blind_pool.state._fsync_directory",
                    side_effect=OSError("synthetic directory fsync failure"),
                ):
                    with self.assertRaises(BlindCanonicalActivationError) as caught:
                        reconcile_blind_canonical_deployment(
                            self.config,
                            case,
                            disposition,
                        )
                self.assertEqual(
                    BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                    caught.exception.category,
                )
                persisted = store.snapshot()
                self.assertIsNone(persisted.reservation)
                if disposition is BlindReconciliationDisposition.CONSUMED:
                    self.assertEqual(before.next_index + 1, persisted.next_index)
                    self.assertEqual(reservation.team_id, persisted.last_consumed_id)
                else:
                    self.assertEqual(before.next_index, persisted.next_index)
                    self.assertEqual(
                        before.last_consumed_id, persisted.last_consumed_id
                    )

    def test_startup_succeeds_after_either_resolution(self):
        for disposition in (
            BlindReconciliationDisposition.CONSUMED,
            BlindReconciliationDisposition.NOT_CONSUMED,
        ):
            with self.subTest(disposition=disposition.value):
                if self.state_path.exists():
                    self.state_path.unlink()
                self.quarantine()
                reconcile_blind_canonical_deployment(
                    self.config,
                    self.current_case(),
                    disposition,
                )
                ready_bytes = self.state_path.read_bytes()
                for command in ("status", "preflight"):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(0, maintenance.main(self.cli_args(command)))
                    self.assertEqual(ready_bytes, self.state_path.read_bytes())
                prepared = prepare_blind_canonical_deployment(self.config)
                self.assertTrue(prepared.owner_held)
                prepared.close()

    def test_normal_startup_still_quarantines_without_reconciliation(self):
        self.quarantine()
        before = self.state_path.read_bytes()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            caught.exception.category,
        )
        self.assertEqual(before, self.state_path.read_bytes())


class TestRawModeCannotReconcile(BlindPoolBagFixture):
    def test_raw_store_rejects_reconciliation_without_mutation(self):
        store = self.store()
        store.initialize_or_load()
        reservation = store.reserve_next(BlindChallengeToken(TOKEN_A))
        store.mark_accept_sent(
            reservation.reservation_id,
            BlindChallengeToken(TOKEN_A),
        )
        state = store.snapshot()
        case = derive_reconciliation_case(state)
        before = self.state_path.read_bytes()
        for disposition in BlindReconciliationDisposition:
            with self.subTest(disposition=disposition.value):
                with self.assertRaises(BlindPoolValidationError) as caught:
                    store.reconcile_accept_sent(case, disposition)
                self.assertEqual(
                    "canonical_reconciliation_required",
                    caught.exception.code,
                )
                self.assertEqual(before, self.state_path.read_bytes())


class TestReconciliationCli(ReconciliationFixture):
    def resolver_process_command(self, case, disposition):
        process_script = """
import pathlib
import sys
from fp.data.blind_pool.maintenance import reconcile_blind_canonical_deployment
from fp.data.blind_pool.reconciliation import BlindReconciliationDisposition
from fp.data.blind_pool.startup import (
    BlindCanonicalActivationError,
    BlindCanonicalStartupConfig,
)

config = BlindCanonicalStartupConfig(
    pathlib.Path(sys.argv[1]),
    pathlib.Path(sys.argv[2]),
    pathlib.Path(sys.argv[3]),
)
try:
    reconcile_blind_canonical_deployment(
        config,
        sys.argv[4],
        BlindReconciliationDisposition(sys.argv[5]),
        owner_timeout_seconds=2.0,
    )
except BlindCanonicalActivationError:
    raise SystemExit(2) from None
"""
        return [
            sys.executable,
            "-B",
            "-c",
            process_script,
            str(self.fixture.private_root),
            str(self.fixture.registry_path),
            str(self.state_path),
            case,
            disposition.value,
        ]

    def run_resolver_processes(self, case, dispositions):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        processes = [
            subprocess.Popen(
                self.resolver_process_command(case, disposition),
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for disposition in dispositions
        ]
        results = [process.communicate(timeout=15) for process in processes]
        return processes, results

    def test_exact_confirmations_are_required_and_cannot_cross(self):
        case = "0" * 64
        parser = maintenance._parser()
        for command, wrong in (
            ("resolve-consumed", "not-consumed"),
            ("resolve-not-consumed", "consumed"),
        ):
            with self.subTest(command=command):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(
                            self.cli_args(command)
                            + ["--case", case, "--confirm", wrong]
                        )
                    with self.assertRaises(SystemExit):
                        parser.parse_args(self.cli_args(command) + ["--case", case])
                    with self.assertRaises(SystemExit):
                        parser.parse_args(self.cli_args(command) + ["--confirm", wrong])

    def test_both_cli_dispositions_have_safe_exact_output(self):
        for command in ("resolve-consumed", "resolve-not-consumed"):
            with self.subTest(command=command):
                if self.state_path.exists():
                    self.state_path.unlink()
                self.quarantine()
                case = self.current_case()
                output = io.StringIO()
                errors = io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    exit_code = maintenance.main(self.resolve_args(command, case))
                rendered = output.getvalue() + errors.getvalue()
                expected = command.removeprefix("resolve-")
                self.assertEqual(0, exit_code)
                self.assertIn("reconciliation resolved: " + expected, rendered)
                self.assertNotIn(TOKEN_A, rendered)
                self.assertNotIn("BL-", rendered)
                self.assertNotIn(str(self.fixture.private_root), rendered)

    def test_owner_contention_blocks_status_and_both_resolvers(self):
        self.quarantine()
        case = self.current_case()
        before = self.state_path.read_bytes()
        owner = acquire_blind_pool_deployment_owner(
            self.state_config(),
            timeout_seconds=0.05,
        )
        try:
            for args in (
                self.cli_args("status"),
                self.resolve_args("resolve-consumed", case),
                self.resolve_args("resolve-not-consumed", case),
            ):
                with self.subTest(command=args[0]):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(2, maintenance.main(args))
                    self.assertEqual(before, self.state_path.read_bytes())
        finally:
            owner.close()

    def test_mismatch_output_and_exception_are_private_and_do_not_echo_case(self):
        self.quarantine()
        current_case = self.current_case()
        wrong_case = "0" * 64 if current_case != "0" * 64 else "1" * 64
        before = self.state_path.read_bytes()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            reconcile_blind_canonical_deployment(
                self.config,
                wrong_case,
                BlindReconciliationDisposition.CONSUMED,
            )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = maintenance.main(
                self.resolve_args("resolve-consumed", wrong_case)
            )
        rendered = output.getvalue() + errors.getvalue()
        self.assertEqual(2, exit_code)
        for forbidden in (
            current_case,
            wrong_case,
            TOKEN_A,
            "BL-",
            str(self.fixture.private_root),
            "metadata_sha256",
            "packed",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_status_and_resolution_commands_are_network_free(self):
        websocket = mock.AsyncMock(side_effect=AssertionError("network"))
        request = mock.Mock(side_effect=AssertionError("network"))
        with mock.patch.object(
            PSWebsocketClient, "create", websocket
        ), mock.patch.object(
            requests.sessions.Session,
            "request",
            request,
        ):
            for command in ("resolve-consumed", "resolve-not-consumed"):
                with self.subTest(command=command):
                    if self.state_path.exists():
                        self.state_path.unlink()
                    self.quarantine()
                    case = self.current_case()
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(0, maintenance.main(self.cli_args("status")))
                        self.assertEqual(
                            0,
                            maintenance.main(self.resolve_args(command, case)),
                        )
        websocket.assert_not_awaited()
        request.assert_not_called()

    def test_status_and_both_resolutions_never_read_full_artifacts(self):
        real_read = canonical_registry._stable_read_bytes

        def reject_full_artifact(path, **options):
            if Path(path).name in {"packed.txt", "team.json"}:
                raise AssertionError("full artifact read")
            return real_read(path, **options)

        with mock.patch.object(
            canonical_registry,
            "_stable_read_bytes",
            side_effect=reject_full_artifact,
        ):
            for command in ("resolve-consumed", "resolve-not-consumed"):
                with self.subTest(command=command):
                    if self.state_path.exists():
                        self.state_path.unlink()
                    self.quarantine()
                    case = self.current_case()
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(0, maintenance.main(self.cli_args("status")))
                        self.assertEqual(
                            0,
                            maintenance.main(self.resolve_args(command, case)),
                        )

    @unittest.skipUnless(os.name == "nt", "Windows process-lock coverage")
    def test_two_processes_racing_same_case_mutate_exactly_once(self):
        store, _reservation = self.quarantine()
        before = store.snapshot()
        case = derive_reconciliation_case(before)
        processes, results = self.run_resolver_processes(
            case,
            (
                BlindReconciliationDisposition.CONSUMED,
                BlindReconciliationDisposition.CONSUMED,
            ),
        )
        return_codes = sorted(process.returncode for process in processes)
        self.assertEqual([0, 2], return_codes, results)
        after = store.snapshot()
        self.assertEqual(before.next_index + 1, after.next_index)
        self.assertIsNone(after.reservation)
        rendered = "".join(part for result in results for part in result)
        self.assertNotIn(TOKEN_A, rendered)
        self.assertNotIn("BL-", rendered)

    @unittest.skipUnless(os.name == "nt", "Windows process-lock coverage")
    def test_opposite_dispositions_racing_same_case_produce_one_complete_result(self):
        store, reservation = self.quarantine()
        before = store.snapshot()
        case = derive_reconciliation_case(before)
        dispositions = (
            BlindReconciliationDisposition.CONSUMED,
            BlindReconciliationDisposition.NOT_CONSUMED,
        )
        processes, results = self.run_resolver_processes(case, dispositions)
        return_codes = [process.returncode for process in processes]
        self.assertEqual([0, 2], sorted(return_codes), results)

        winner = dispositions[return_codes.index(0)]
        after = store.snapshot()
        self.assertIsNone(after.reservation)
        if winner is BlindReconciliationDisposition.CONSUMED:
            self.assertEqual(before.next_index + 1, after.next_index)
            self.assertEqual(reservation.team_id, after.last_consumed_id)
        else:
            self.assertEqual(before.next_index, after.next_index)
            self.assertEqual(before.last_consumed_id, after.last_consumed_id)
        rendered = "".join(part for result in results for part in result)
        self.assertNotIn(TOKEN_A, rendered)
        self.assertNotIn("BL-", rendered)


if __name__ == "__main__":
    unittest.main()
