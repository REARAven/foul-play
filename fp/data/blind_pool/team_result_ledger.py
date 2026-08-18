"""Strict append-only authoritative result ledger for team-era Blind Ladder."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Callable, NoReturn

from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import is_valid_opaque_team_id, is_valid_player_team_id
from .team_persistence import (
    TeamStateFileConfig,
    stable_read_bytes,
    validate_team_state_file_config,
    write_atomic,
)
from .team_rating import (
    TEAM_OUTCOME_A_LOSS,
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_NO_RESULT,
    TEAM_OUTCOME_TIE,
    TEAM_RESULT_OUTCOMES,
    BlindTeamBattleResult,
)


TEAM_RESULT_LEDGER_SCHEMA_VERSION = 1
TEAM_RESULT_RECORD_VERSION = 1
TEAM_RESULT_LEDGER_PATH_ENV = "TUGS_BLIND_TEAM_RESULT_LEDGER"
TEAM_PENDING_ROOM_VALIDATED = "room_validated"
TEAM_PENDING_AWAITING_TERMINAL = "awaiting_terminal"
TEAM_PENDING_PHASES = frozenset(
    {TEAM_PENDING_ROOM_VALIDATED, TEAM_PENDING_AWAITING_TERMINAL}
)
TEAM_RESOLUTION_SHOWDOWN = "showdown_terminal"
TEAM_RESOLUTION_OPERATOR = "operator_recovery"
TEAM_RESOLUTION_SOURCES = frozenset(
    {TEAM_RESOLUTION_SHOWDOWN, TEAM_RESOLUTION_OPERATOR}
)
TEAM_BATTLE_ID_DOMAIN = b"TUGS-BLIND-TEAM-BATTLE-v1"
TEAM_RESULT_RECORD_DOMAIN = b"TUGS-BLIND-TEAM-RESULT-RECORD-v1"
TEAM_RESULT_RECOVERY_DOMAIN = b"TUGS-BLIND-TEAM-RESULT-RECOVERY-v1"
ZERO_TEAM_RESULT_HASH = "0" * 64

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SHOWDOWN_ID = re.compile(r"^[a-z0-9]{1,18}$")
_ROOM_ID = re.compile(r"^battle-(?P<format_id>[a-z0-9]+)-[a-z0-9-]+$")
_FORMAT_ID = re.compile(r"^[a-z0-9]+$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_TOP_FIELDS = frozenset({"schema_version", "completed_results", "pending_result"})
_IDENTITY_FIELDS = frozenset(
    {
        "battle_id",
        "player_account_id",
        "bot_account_id",
        "player_team_id",
        "bot_team_id",
        "reservation_id",
        "room_id",
        "format_id",
        "registry_fingerprint",
        "correlated_at",
    }
)
_PENDING_FIELDS = _IDENTITY_FIELDS | frozenset({"record_version", "phase"})
_COMPLETED_FIELDS = _IDENTITY_FIELDS | frozenset(
    {
        "record_version",
        "sequence",
        "resolved_at",
        "outcome",
        "resolution_source",
        "previous_record_hash",
        "record_hash",
    }
)


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamResultLedgerConfig:
    ledger_path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    registry_path: Path = field(repr=False)
    selection_state_path: Path = field(repr=False)
    rating_state_path: Path | None = field(default=None, repr=False)
    public_registry_path: Path | None = field(default=None, repr=False)
    repository_root: Path | None = field(default=None, repr=False)

    @property
    def lock_path(self) -> Path:
        return self.ledger_path.with_name(self.ledger_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindTeamResultLedgerConfig(configured=True)"


def validate_team_result_ledger_config(
    ledger_path: str | Path,
    *,
    private_root: str | Path,
    registry_path: str | Path,
    selection_state_path: str | Path,
    rating_state_path: str | Path | None = None,
    public_registry_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> BlindTeamResultLedgerConfig:
    collisions = [Path(registry_path), Path(selection_state_path)]
    if rating_state_path is not None:
        collisions.append(Path(rating_state_path))
    if public_registry_path is not None:
        collisions.append(Path(public_registry_path))
    validated = validate_team_state_file_config(
        ledger_path,
        private_root=private_root,
        collision_paths=tuple(collisions),
        code_prefix="team_result",
        repository_root=repository_root,
    )
    return BlindTeamResultLedgerConfig(
        validated.path,
        validated.private_root,
        Path(registry_path).resolve(strict=False),
        Path(selection_state_path).resolve(strict=False),
        None
        if rating_state_path is None
        else Path(rating_state_path).resolve(strict=False),
        None
        if public_registry_path is None
        else Path(public_registry_path).resolve(strict=False),
        validated.repository_root,
    )


def _file_config(config: BlindTeamResultLedgerConfig) -> TeamStateFileConfig:
    collisions = [config.registry_path, config.selection_state_path]
    if config.rating_state_path is not None:
        collisions.append(config.rating_state_path)
    if config.public_registry_path is not None:
        collisions.append(config.public_registry_path)
    return validate_team_state_file_config(
        config.ledger_path,
        private_root=config.private_root,
        collision_paths=tuple(collisions),
        code_prefix="team_result",
        repository_root=config.repository_root,
    )


def _framed_digest(domain: bytes, fields: tuple[tuple[str, str], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for label, value in fields:
        label_bytes = label.encode("ascii")
        value_bytes = value.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def derive_blind_team_battle_id(
    *,
    reservation_id: str,
    room_id: str,
    player_account_id: str,
    bot_account_id: str,
    player_team_id: str,
    bot_team_id: str,
    format_id: str,
    registry_fingerprint: str,
) -> str:
    """Derive a deterministic private identity with no challenge-token material."""

    values = {
        "reservation_id": reservation_id,
        "room_id": room_id,
        "player_account_id": player_account_id,
        "bot_account_id": bot_account_id,
        "player_team_id": player_team_id,
        "bot_team_id": bot_team_id,
        "format_id": format_id,
        "registry_fingerprint": registry_fingerprint,
    }
    room_match = _ROOM_ID.fullmatch(room_id) if isinstance(room_id, str) else None
    if (
        not isinstance(reservation_id, str)
        or _HEX_32.fullmatch(reservation_id) is None
        or room_match is None
        or not isinstance(player_account_id, str)
        or _SHOWDOWN_ID.fullmatch(player_account_id) is None
        or not isinstance(bot_account_id, str)
        or _SHOWDOWN_ID.fullmatch(bot_account_id) is None
        or player_account_id == bot_account_id
        or not is_valid_player_team_id(player_team_id)
        or not is_valid_opaque_team_id(bot_team_id)
        or player_team_id == bot_team_id
        or not isinstance(format_id, str)
        or _FORMAT_ID.fullmatch(format_id) is None
        or format_id != "gen9tugs"
        or room_match.group("format_id") != format_id
        or not isinstance(registry_fingerprint, str)
        or _HEX_64.fullmatch(registry_fingerprint) is None
    ):
        _fail("team_result_identity_invalid", "Team result identity is invalid")
    return _framed_digest(
        TEAM_BATTLE_ID_DOMAIN,
        tuple((label, values[label]) for label in sorted(values)),
    )


def _format_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail("team_result_clock_invalid", "Team result clock is invalid")
    try:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (OverflowError, ValueError):
        _fail("team_result_clock_invalid", "Team result clock is invalid")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        _fail("team_result_timestamp_invalid", "Team result timestamp is invalid")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        _fail("team_result_timestamp_invalid", "Team result timestamp is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamPendingBattleResult:
    record_version: int
    phase: str
    battle_id: str = field(repr=False)
    player_account_id: str = field(repr=False)
    bot_account_id: str = field(repr=False)
    player_team_id: str = field(repr=False)
    bot_team_id: str = field(repr=False)
    reservation_id: str = field(repr=False)
    room_id: str = field(repr=False)
    format_id: str
    registry_fingerprint: str = field(repr=False)
    correlated_at: str

    def __repr__(self) -> str:
        return "BlindTeamPendingBattleResult(phase={!r})".format(self.phase)


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamCompletedBattleResult:
    record_version: int
    sequence: int
    battle_id: str = field(repr=False)
    player_account_id: str = field(repr=False)
    bot_account_id: str = field(repr=False)
    player_team_id: str = field(repr=False)
    bot_team_id: str = field(repr=False)
    reservation_id: str = field(repr=False)
    room_id: str = field(repr=False)
    format_id: str
    registry_fingerprint: str = field(repr=False)
    correlated_at: str
    resolved_at: str
    outcome: str
    resolution_source: str
    previous_record_hash: str = field(repr=False)
    record_hash: str = field(repr=False)

    def to_rating_result(self) -> BlindTeamBattleResult:
        return BlindTeamBattleResult(
            self.sequence,
            self.previous_record_hash,
            self.record_hash,
            self.player_team_id,
            self.bot_team_id,
            self.outcome,
        )

    def __repr__(self) -> str:
        return "BlindTeamCompletedBattleResult(sequence={!r}, outcome={!r})".format(
            self.sequence, self.outcome
        )


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamResultLedgerState:
    schema_version: int
    completed_results: tuple[BlindTeamCompletedBattleResult, ...] = field(repr=False)
    pending_result: BlindTeamPendingBattleResult | None = field(repr=False)

    @property
    def completed_count(self) -> int:
        return len(self.completed_results)

    @property
    def pending_count(self) -> int:
        return int(self.pending_result is not None)

    def __repr__(self) -> str:
        return "BlindTeamResultLedgerState(completed_count={!r}, pending_count={!r})".format(
            self.completed_count, self.pending_count
        )


def _pending_document(value: BlindTeamPendingBattleResult) -> dict[str, Any]:
    return {name: getattr(value, name) for name in _PENDING_FIELDS}


def _completed_document(value: BlindTeamCompletedBattleResult) -> dict[str, Any]:
    return {name: getattr(value, name) for name in _COMPLETED_FIELDS}


def _state_document(state: BlindTeamResultLedgerState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "completed_results": [
            _completed_document(item) for item in state.completed_results
        ],
        "pending_result": None
        if state.pending_result is None
        else _pending_document(state.pending_result),
    }


def _validate_exact(value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("team_result_document_invalid", "Team result document is invalid")
    if set(value) != fields:
        _fail("team_result_fields_invalid", "Team result fields are invalid")
    return value


def _validate_identity(value: dict[str, Any]) -> None:
    expected = derive_blind_team_battle_id(
        reservation_id=value["reservation_id"],
        room_id=value["room_id"],
        player_account_id=value["player_account_id"],
        bot_account_id=value["bot_account_id"],
        player_team_id=value["player_team_id"],
        bot_team_id=value["bot_team_id"],
        format_id=value["format_id"],
        registry_fingerprint=value["registry_fingerprint"],
    )
    if value["battle_id"] != expected:
        _fail("team_result_battle_id_mismatch", "Team result identity is inconsistent")
    _parse_timestamp(value["correlated_at"])


def _record_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("record_hash", None)
    canonical = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return _framed_digest(TEAM_RESULT_RECORD_DOMAIN, (("record", canonical),))


def validate_team_result_ledger_document(document: Any) -> BlindTeamResultLedgerState:
    root = _validate_exact(document, _TOP_FIELDS)
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != TEAM_RESULT_LEDGER_SCHEMA_VERSION
    ):
        _fail("team_result_schema_unsupported", "Team result schema is unsupported")
    if not isinstance(root["completed_results"], list):
        _fail("team_result_history_invalid", "Team result history is invalid")
    completed: list[BlindTeamCompletedBattleResult] = []
    previous_hash = ZERO_TEAM_RESULT_HASH
    identities: set[tuple[str, str, str]] = set()
    for sequence, raw in enumerate(root["completed_results"], start=1):
        value = _validate_exact(raw, _COMPLETED_FIELDS)
        if (
            type(value["record_version"]) is not int
            or value["record_version"] != TEAM_RESULT_RECORD_VERSION
        ):
            _fail(
                "team_result_record_version_unsupported",
                "Team result record version is unsupported",
            )
        if type(value["sequence"]) is not int or value["sequence"] != sequence:
            _fail("team_result_sequence_invalid", "Team result sequence is invalid")
        _validate_identity(value)
        correlated = _parse_timestamp(value["correlated_at"])
        resolved = _parse_timestamp(value["resolved_at"])
        if resolved < correlated:
            _fail(
                "team_result_timestamp_order_invalid",
                "Team result timestamps are inconsistent",
            )
        if value["outcome"] not in TEAM_RESULT_OUTCOMES:
            _fail("team_result_outcome_invalid", "Team result outcome is invalid")
        if value["resolution_source"] not in TEAM_RESOLUTION_SOURCES:
            _fail("team_result_resolution_invalid", "Team result resolution is invalid")
        if (
            value["outcome"] == TEAM_OUTCOME_NO_RESULT
            and value["resolution_source"] != TEAM_RESOLUTION_OPERATOR
        ):
            _fail(
                "team_result_disposition_invalid",
                "No-result requires operator recovery",
            )
        if value["previous_record_hash"] != previous_hash:
            _fail(
                "team_result_history_chain_invalid",
                "Team result history is inconsistent",
            )
        if value["record_hash"] != _record_hash(value):
            _fail("team_result_record_hash_invalid", "Team result hash is invalid")
        identity = (value["battle_id"], value["room_id"], value["reservation_id"])
        if any(
            item in {part[index] for part in identities}
            for index, item in enumerate(identity)
        ):
            _fail(
                "team_result_history_identity_reused", "Team result identity is reused"
            )
        identities.add(identity)
        record = BlindTeamCompletedBattleResult(**value)
        record.to_rating_result()
        completed.append(record)
        previous_hash = record.record_hash
    pending = None
    if root["pending_result"] is not None:
        value = _validate_exact(root["pending_result"], _PENDING_FIELDS)
        if (
            type(value["record_version"]) is not int
            or value["record_version"] != TEAM_RESULT_RECORD_VERSION
        ):
            _fail(
                "team_result_record_version_unsupported",
                "Team result record version is unsupported",
            )
        if value["phase"] not in TEAM_PENDING_PHASES:
            _fail("team_result_pending_phase_invalid", "Team result phase is invalid")
        _validate_identity(value)
        pending = BlindTeamPendingBattleResult(**value)
        if any(
            pending.battle_id == item.battle_id
            or pending.room_id == item.room_id
            or pending.reservation_id == item.reservation_id
            for item in completed
        ):
            _fail("team_result_pending_identity_reused", "Pending identity is reused")
    return BlindTeamResultLedgerState(
        TEAM_RESULT_LEDGER_SCHEMA_VERSION, tuple(completed), pending
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _decode(raw: bytes) -> BlindTeamResultLedgerState:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("team_result_encoding_invalid", "Team result ledger has a BOM")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except _DuplicateJsonFieldError:
        _fail("team_result_duplicate_field", "Team result ledger has duplicate fields")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("team_result_json_invalid", "Team result ledger is malformed")
    return validate_team_result_ledger_document(document)


def _encode(state: BlindTeamResultLedgerState) -> bytes:
    validated = validate_team_result_ledger_document(_state_document(state))
    return (
        json.dumps(
            _state_document(validated),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _recovery_case_for(state: BlindTeamResultLedgerState) -> str:
    if state.pending_result is None:
        _fail(
            "team_result_recovery_not_applicable",
            "Team result recovery is not applicable",
        )
    previous = (
        state.completed_results[-1].record_hash
        if state.completed_results
        else ZERO_TEAM_RESULT_HASH
    )
    pending = json.dumps(
        _pending_document(state.pending_result),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _framed_digest(
        TEAM_RESULT_RECOVERY_DOMAIN,
        (("history_hash", previous), ("pending", pending)),
    )


class BlindTeamResultLedgerStore:
    """Independent one-lock transactional store for authoritative team results."""

    def __init__(
        self,
        config: BlindTeamResultLedgerConfig,
        *,
        lock_timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, BlindTeamResultLedgerConfig):
            _fail("team_result_config_invalid", "Team result configuration is invalid")
        self._config = config
        self._lock_timeout = lock_timeout_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._history_baseline: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return "BlindTeamResultLedgerStore(configured=True)"

    def _load_unlocked(self, config: TeamStateFileConfig) -> BlindTeamResultLedgerState:
        state = _decode(stable_read_bytes(config))
        hashes = tuple(item.record_hash for item in state.completed_results)
        if hashes[: len(self._history_baseline)] != self._history_baseline:
            _fail("team_result_history_changed", "Team result history changed")
        self._history_baseline = hashes
        return state

    def initialize_empty(self) -> BlindTeamResultLedgerState:
        config = _file_config(self._config)
        state = BlindTeamResultLedgerState(TEAM_RESULT_LEDGER_SCHEMA_VERSION, (), None)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            if config.path.exists() or config.path.is_symlink():
                existing = self._load_unlocked(config)
                if existing.completed_count or existing.pending_count:
                    _fail(
                        "team_result_ledger_not_empty",
                        "Team result ledger is not empty",
                    )
                return existing
            write_atomic(config, _encode(state), replace_existing=False)
        return state

    def load(self) -> BlindTeamResultLedgerState:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            return self._load_unlocked(config)

    def require_ready(self) -> BlindTeamResultLedgerState:
        state = self.load()
        if state.pending_result is not None:
            _fail("team_result_recovery_required", "Team result recovery is required")
        return state

    def create_pending_intent(
        self,
        *,
        player_account_id: str,
        bot_account_id: str,
        player_team_id: str,
        bot_team_id: str,
        reservation_id: str,
        room_id: str,
        format_id: str,
        registry_fingerprint: str,
    ) -> BlindTeamPendingBattleResult:
        battle_id = derive_blind_team_battle_id(
            reservation_id=reservation_id,
            room_id=room_id,
            player_account_id=player_account_id,
            bot_account_id=bot_account_id,
            player_team_id=player_team_id,
            bot_team_id=bot_team_id,
            format_id=format_id,
            registry_fingerprint=registry_fingerprint,
        )
        candidate = BlindTeamPendingBattleResult(
            TEAM_RESULT_RECORD_VERSION,
            TEAM_PENDING_ROOM_VALIDATED,
            battle_id,
            player_account_id,
            bot_account_id,
            player_team_id,
            bot_team_id,
            reservation_id,
            room_id,
            format_id,
            registry_fingerprint,
            _format_timestamp(self._clock()),
        )
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            for record in state.completed_results:
                if (
                    record.battle_id == battle_id
                    or record.room_id == room_id
                    or record.reservation_id == reservation_id
                ):
                    _fail(
                        "team_result_identity_conflict",
                        "Team result identity conflicts",
                    )
            if state.pending_result is not None:
                existing = state.pending_result
                comparable = (
                    "battle_id",
                    "player_account_id",
                    "bot_account_id",
                    "player_team_id",
                    "bot_team_id",
                    "reservation_id",
                    "room_id",
                    "format_id",
                    "registry_fingerprint",
                )
                if all(
                    getattr(existing, name) == getattr(candidate, name)
                    for name in comparable
                ):
                    return existing
                _fail("team_result_pending_exists", "A team result is already pending")
            updated = BlindTeamResultLedgerState(
                state.schema_version, state.completed_results, candidate
            )
            write_atomic(config, _encode(updated), replace_existing=True)
            return candidate

    def mark_selection_committed(self, battle_id: str) -> BlindTeamPendingBattleResult:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            pending = state.pending_result
            if pending is None or pending.battle_id != battle_id:
                _fail(
                    "team_result_pending_mismatch", "Pending team result does not match"
                )
            if pending.phase == TEAM_PENDING_AWAITING_TERMINAL:
                return pending
            if pending.phase != TEAM_PENDING_ROOM_VALIDATED:
                _fail(
                    "team_result_pending_phase_invalid",
                    "Pending team result phase is invalid",
                )
            pending = replace(pending, phase=TEAM_PENDING_AWAITING_TERMINAL)
            write_atomic(
                config,
                _encode(replace(state, pending_result=pending)),
                replace_existing=True,
            )
            return pending

    def finalize_terminal(
        self, battle_id: str, outcome: str
    ) -> BlindTeamCompletedBattleResult:
        if outcome not in {TEAM_OUTCOME_A_WIN, TEAM_OUTCOME_A_LOSS, TEAM_OUTCOME_TIE}:
            _fail(
                "team_result_terminal_outcome_invalid",
                "Terminal team outcome is invalid",
            )
        return self._finalize(battle_id, outcome, TEAM_RESOLUTION_SHOWDOWN)

    def _finalize(
        self, battle_id: str, outcome: str, source: str
    ) -> BlindTeamCompletedBattleResult:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            existing = next(
                (
                    item
                    for item in state.completed_results
                    if item.battle_id == battle_id
                ),
                None,
            )
            if existing is not None:
                if existing.outcome == outcome and existing.resolution_source == source:
                    return existing
                _fail(
                    "team_result_outcome_conflict", "Completed team result is immutable"
                )
            pending = state.pending_result
            if pending is None or pending.battle_id != battle_id:
                _fail(
                    "team_result_pending_mismatch", "Pending team result does not match"
                )
            if (
                source == TEAM_RESOLUTION_SHOWDOWN
                and pending.phase != TEAM_PENDING_AWAITING_TERMINAL
            ):
                _fail(
                    "team_result_pending_phase_invalid", "Terminal result is premature"
                )
            previous_hash = (
                state.completed_results[-1].record_hash
                if state.completed_results
                else ZERO_TEAM_RESULT_HASH
            )
            document = {
                **_pending_document(pending),
                "sequence": state.completed_count + 1,
                "resolved_at": _format_timestamp(self._clock()),
                "outcome": outcome,
                "resolution_source": source,
                "previous_record_hash": previous_hash,
                "record_hash": "",
            }
            document.pop("phase")
            document["record_hash"] = _record_hash(document)
            validated = validate_team_result_ledger_document(
                {
                    "schema_version": TEAM_RESULT_LEDGER_SCHEMA_VERSION,
                    "completed_results": [
                        *[
                            _completed_document(item)
                            for item in state.completed_results
                        ],
                        document,
                    ],
                    "pending_result": None,
                }
            )
            record = validated.completed_results[-1]
            write_atomic(config, _encode(validated), replace_existing=True)
            self._history_baseline = tuple(
                item.record_hash for item in validated.completed_results
            )
            return record

    def recovery_case(self) -> str:
        state = self.load()
        return _recovery_case_for(state)

    def resolve_pending(
        self, recovery_case: str, outcome: str
    ) -> BlindTeamCompletedBattleResult:
        if outcome not in TEAM_RESULT_OUTCOMES:
            _fail("team_result_outcome_invalid", "Recovery outcome is invalid")
        if (
            not isinstance(recovery_case, str)
            or _HEX_64.fullmatch(recovery_case) is None
        ):
            _fail("team_result_recovery_case_invalid", "Recovery case is invalid")
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            expected = _recovery_case_for(state)
            if not hmac.compare_digest(recovery_case, expected):
                _fail("team_result_recovery_case_mismatch", "Recovery case is stale")
            pending = state.pending_result
            assert pending is not None
            previous_hash = (
                state.completed_results[-1].record_hash
                if state.completed_results
                else ZERO_TEAM_RESULT_HASH
            )
            document = {
                **_pending_document(pending),
                "sequence": state.completed_count + 1,
                "resolved_at": _format_timestamp(self._clock()),
                "outcome": outcome,
                "resolution_source": TEAM_RESOLUTION_OPERATOR,
                "previous_record_hash": previous_hash,
                "record_hash": "",
            }
            document.pop("phase")
            document["record_hash"] = _record_hash(document)
            updated = validate_team_result_ledger_document(
                {
                    "schema_version": TEAM_RESULT_LEDGER_SCHEMA_VERSION,
                    "completed_results": [
                        *[
                            _completed_document(item)
                            for item in state.completed_results
                        ],
                        document,
                    ],
                    "pending_result": None,
                }
            )
            write_atomic(config, _encode(updated), replace_existing=True)
            self._history_baseline = tuple(
                item.record_hash for item in updated.completed_results
            )
            return updated.completed_results[-1]
