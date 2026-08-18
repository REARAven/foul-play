from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest

from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.locking import BlindPoolStateLock
from fp.data.blind_pool.result_ledger import (
    OUTCOME_NO_RESULT,
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
    PENDING_AWAITING_TERMINAL,
    PENDING_ROOM_VALIDATED,
    RESULT_LEDGER_SCHEMA_VERSION,
    BlindResultLedgerStore,
    derive_blind_battle_id,
    validate_result_ledger_config,
    validate_result_ledger_document,
)


PRIVATE_SENTINEL = "PHASE6A-SYNTHETIC-SEALED-SENTINEL"
CHALLENGE_TOKEN_SENTINEL = "f" * 32
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


class ResultLedgerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="result-ledger-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private_root = self.root / "deployment"
        self.private_root.mkdir()
        self.registry_path = self.private_root / "registry.json"
        self.registry_path.write_text("{}\n", encoding="utf-8")
        self.state_path = self.private_root / "state.json"
        self.ladder_root = self.root / "ladder"
        self.ladder_root.mkdir()
        self.ledger_path = self.ladder_root / "results.json"
        self.clock_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.config = validate_result_ledger_config(
            self.ledger_path,
            private_root=self.private_root,
            registry_path=self.registry_path,
            selection_state_path=self.state_path,
        )
        self.store = BlindResultLedgerStore(
            self.config,
            lock_timeout_seconds=0.1,
            clock=lambda: self.clock_value,
        )

    def advance_clock(self):
        self.clock_value += timedelta(seconds=1)

    def initialize(self):
        return self.store.initialize_empty()

    def intent_kwargs(self, *, number=1, fingerprint=FINGERPRINT_A):
        return {
            "player_id": "syntheticplayer",
            "bot_id": "syntheticbot",
            "team_id": "BL-{:03d}-v1".format(number),
            "reservation_id": format(number, "032x"),
            "room_id": "battle-gen9tugs-{}".format(400 + number),
            "format_id": "gen9tugs",
            "registry_fingerprint": fingerprint,
        }

    def create_pending(self, **overrides):
        values = self.intent_kwargs()
        values.update(overrides)
        return self.store.create_pending_intent(**values)

    def create_awaiting(self, **overrides):
        pending = self.create_pending(**overrides)
        return self.store.mark_selection_committed(pending.battle_id)

    def assert_code(self, code, callback):
        with self.assertRaises(BlindPoolValidationError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)
        rendered = str(caught.exception) + repr(caught.exception)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertNotIn(str(self.root), rendered)


class ResultLedgerValidationTests(ResultLedgerFixture):
    def test_empty_ledger_initializes_strict_utf8_lf_and_is_idempotent(self):
        state = self.initialize()
        self.assertEqual(RESULT_LEDGER_SCHEMA_VERSION, state.schema_version)
        self.assertEqual(0, state.completed_count)
        self.assertEqual(0, state.pending_count)
        raw = self.ledger_path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)
        self.assertEqual(state, self.store.initialize_empty())

    def test_strict_schema_rejects_missing_extra_and_unsupported_fields(self):
        valid = {
            "schema_version": 1,
            "completed_results": [],
            "pending_result": None,
        }
        for mutation, code in (
            (
                lambda value: value.pop("pending_result"),
                "result_missing_required_field",
            ),
            (lambda value: value.update(extra=True), "result_unexpected_field"),
            (lambda value: value.update(schema_version=2), "result_schema_unsupported"),
        ):
            document = dict(valid)
            mutation(document)
            with self.subTest(code=code):
                self.assert_code(
                    code,
                    lambda document=document: validate_result_ledger_document(document),
                )

    def test_duplicate_json_keys_malformed_utf8_json_and_bom_are_rejected(self):
        self.initialize()
        cases = (
            (b'{"schema_version":1,"schema_version":1}', "result_duplicate_field"),
            (b"\xff", "result_encoding_invalid"),
            (b"{", "result_json_invalid"),
            (b"\xef\xbb\xbf{}", "result_encoding_invalid"),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                self.ledger_path.write_bytes(payload)
                fresh = BlindResultLedgerStore(self.config, lock_timeout_seconds=0.1)
                self.assert_code(code, fresh.load)

    def test_invalid_pending_outcome_identity_and_sequence_are_rejected(self):
        self.initialize()
        pending = self.create_pending()
        document = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        document["pending_result"]["battle_id"] = "0" * 64
        self.assert_code(
            "result_battle_id_mismatch",
            lambda: validate_result_ledger_document(document),
        )
        self.store.mark_selection_committed(pending.battle_id)
        self.advance_clock()
        self.store.finalize_terminal(pending.battle_id, OUTCOME_PLAYER_WIN)
        document = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        document["completed_results"][0]["sequence"] = 2
        self.assert_code(
            "result_sequence_invalid",
            lambda: validate_result_ledger_document(document),
        )
        document = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        document["completed_results"][0]["outcome"] = "unsupported"
        self.assert_code(
            "result_outcome_invalid",
            lambda: validate_result_ledger_document(document),
        )

    def test_modified_or_reordered_hash_chained_history_is_rejected(self):
        self.initialize()
        first = self.create_awaiting()
        self.advance_clock()
        self.store.finalize_terminal(first.battle_id, OUTCOME_PLAYER_WIN)
        second = self.create_awaiting(
            team_id="BL-002-v1",
            reservation_id="2" * 32,
            room_id="battle-gen9tugs-402",
        )
        self.advance_clock()
        self.store.finalize_terminal(second.battle_id, OUTCOME_PLAYER_LOSS)
        original = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        modified = json.loads(json.dumps(original))
        modified["completed_results"][0]["outcome"] = OUTCOME_TIE
        self.assert_code(
            "result_record_hash_invalid",
            lambda: validate_result_ledger_document(modified),
        )
        reordered = json.loads(json.dumps(original))
        reordered["completed_results"].reverse()
        self.assert_code(
            "result_sequence_invalid",
            lambda: validate_result_ledger_document(reordered),
        )

    def test_repository_and_deployment_paths_are_rejected(self):
        repository_ledger = Path.cwd() / "synthetic-result-ledger.json"
        self.assert_code(
            "result_path_not_external",
            lambda: validate_result_ledger_config(
                repository_ledger,
                private_root=self.private_root,
                registry_path=self.registry_path,
                selection_state_path=self.state_path,
            ),
        )
        self.assert_code(
            "result_path_in_deployment",
            lambda: validate_result_ledger_config(
                self.private_root / "results.json",
                private_root=self.private_root,
                registry_path=self.registry_path,
                selection_state_path=self.state_path,
            ),
        )

    def test_symlink_ledger_is_rejected_when_supported(self):
        target = self.ladder_root / "target.json"
        target.write_text("{}", encoding="utf-8")
        link = self.ladder_root / "link.json"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("symlink creation is unavailable")
        self.assert_code(
            "result_path_unsafe",
            lambda: validate_result_ledger_config(
                link,
                private_root=self.private_root,
                registry_path=self.registry_path,
                selection_state_path=self.state_path,
            ),
        )

    def test_dedicated_lock_serializes_concurrent_access(self):
        self.initialize()
        observed = []

        def blocked_load():
            try:
                BlindResultLedgerStore(
                    self.config,
                    lock_timeout_seconds=0.02,
                ).load()
            except BlindPoolValidationError as error:
                observed.append(error.code)

        with BlindPoolStateLock(self.config.lock_path, timeout_seconds=0.1):
            thread = threading.Thread(target=blocked_load)
            thread.start()
            thread.join()
        self.assertEqual(["state_lock_timeout"], observed)


class ResultLedgerTransactionTests(ResultLedgerFixture):
    def setUp(self):
        super().setUp()
        self.initialize()

    def test_battle_id_is_deterministic_framed_and_token_independent(self):
        values = self.intent_kwargs()
        first = derive_blind_battle_id(
            **{
                key: values[key]
                for key in (
                    "reservation_id",
                    "room_id",
                    "player_id",
                    "bot_id",
                    "team_id",
                    "registry_fingerprint",
                )
            }
        )
        second = self.create_pending().battle_id
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))
        self.assertNotIn(CHALLENGE_TOKEN_SENTINEL, self.ledger_path.read_text())

    def test_identical_pending_creation_is_idempotent_conflict_fails(self):
        first = self.create_pending()
        self.advance_clock()
        second = self.create_pending()
        self.assertEqual(first, second)
        self.assert_code(
            "result_room_identity_conflict",
            lambda: self.create_pending(
                team_id="BL-002-v1",
                reservation_id="2" * 32,
            ),
        )
        self.assert_code(
            "result_reservation_identity_conflict",
            lambda: self.create_pending(
                team_id="BL-002-v1",
                room_id="battle-gen9tugs-999",
            ),
        )

    def test_pending_phase_transition_is_durable_and_idempotent(self):
        pending = self.create_pending()
        self.assertEqual(PENDING_ROOM_VALIDATED, pending.phase)
        awaiting = self.store.mark_selection_committed(pending.battle_id)
        self.assertEqual(PENDING_AWAITING_TERMINAL, awaiting.phase)
        self.assertEqual(
            awaiting, self.store.mark_selection_committed(pending.battle_id)
        )
        reloaded = BlindResultLedgerStore(self.config).load()
        self.assertEqual(PENDING_AWAITING_TERMINAL, reloaded.pending_result.phase)

    def test_terminal_outcomes_append_once_and_conflict_fails(self):
        pending = self.create_awaiting()
        self.advance_clock()
        first = self.store.finalize_terminal(pending.battle_id, OUTCOME_PLAYER_WIN)
        duplicate = self.store.finalize_terminal(
            pending.battle_id,
            OUTCOME_PLAYER_WIN,
        )
        self.assertEqual(first, duplicate)
        self.assertEqual(1, self.store.load().completed_count)
        self.assert_code(
            "result_outcome_conflict",
            lambda: self.store.finalize_terminal(
                pending.battle_id,
                OUTCOME_PLAYER_LOSS,
            ),
        )

    def test_completed_identity_cannot_return_to_pending(self):
        pending = self.create_awaiting()
        self.store.finalize_terminal(pending.battle_id, OUTCOME_TIE)
        self.assert_code("result_battle_completed", self.create_pending)

    def test_challenge_token_and_synthetic_team_body_are_never_serialized(self):
        self.create_pending()
        rendered = self.ledger_path.read_text(encoding="utf-8")
        self.assertNotIn(CHALLENGE_TOKEN_SENTINEL, rendered)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertEqual(1, rendered.count("BL-001-v1"))

    def test_unresolved_pending_blocks_ready_check(self):
        self.create_pending()
        self.assert_code("result_recovery_required", self.store.require_ready)

    def test_historical_old_fingerprint_survives_new_current_identity(self):
        old = self.create_awaiting(registry_fingerprint=FINGERPRINT_A)
        self.store.finalize_terminal(old.battle_id, OUTCOME_PLAYER_WIN)
        current = self.create_pending(
            team_id="BL-002-v1",
            reservation_id="2" * 32,
            room_id="battle-gen9tugs-402",
            registry_fingerprint=FINGERPRINT_B,
        )
        state = self.store.load()
        self.assertEqual(FINGERPRINT_A, state.completed_results[0].registry_fingerprint)
        self.assertEqual(FINGERPRINT_B, current.registry_fingerprint)


class ResultLedgerRecoveryTests(ResultLedgerFixture):
    def test_recovery_case_is_deterministic_and_stale_case_is_rejected(self):
        self.initialize()
        self.create_pending()
        first = self.store.recovery_case()
        second = BlindResultLedgerStore(self.config).recovery_case()
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))
        self.assert_code(
            "result_recovery_case_mismatch",
            lambda: self.store.resolve_pending("0" * 64, OUTCOME_NO_RESULT),
        )

    def test_all_explicit_recovery_dispositions_are_atomic_and_immutable(self):
        for outcome in (
            OUTCOME_PLAYER_WIN,
            OUTCOME_PLAYER_LOSS,
            OUTCOME_TIE,
            OUTCOME_NO_RESULT,
        ):
            with self.subTest(outcome=outcome):
                self.tearDown()
                self.setUp()
                self.initialize()
                self.create_pending()
                recovery_case = self.store.recovery_case()
                record = self.store.resolve_pending(recovery_case, outcome)
                state = self.store.load()
                self.assertEqual(outcome, record.outcome)
                self.assertEqual(1, state.completed_count)
                self.assertEqual(0, state.pending_count)
                self.assert_code(
                    "result_recovery_not_applicable",
                    self.store.recovery_case,
                )

    def test_no_result_cannot_be_changed_to_a_terminal_outcome(self):
        self.initialize()
        pending = self.create_pending()
        record = self.store.resolve_pending(
            self.store.recovery_case(),
            OUTCOME_NO_RESULT,
        )
        self.assertEqual(pending.battle_id, record.battle_id)
        self.assert_code(
            "result_outcome_conflict",
            lambda: self.store.finalize_terminal(
                pending.battle_id,
                OUTCOME_PLAYER_WIN,
            ),
        )


if __name__ == "__main__":
    unittest.main()
