"""Strict, durable, privacy-safe Blind Ladder battle-result persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, NoReturn

from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import is_valid_opaque_team_id


RESULT_LEDGER_SCHEMA_VERSION = 1
RESULT_RECORD_VERSION = 1
RESULT_LEDGER_PATH_ENV = "TUGS_BLIND_RESULT_LEDGER"

PENDING_ROOM_VALIDATED = "room_validated"
PENDING_AWAITING_TERMINAL = "awaiting_terminal"
PENDING_PHASES = frozenset({PENDING_ROOM_VALIDATED, PENDING_AWAITING_TERMINAL})

OUTCOME_PLAYER_WIN = "player_win"
OUTCOME_PLAYER_LOSS = "player_loss"
OUTCOME_TIE = "tie"
OUTCOME_NO_RESULT = "no_result"
RESULT_OUTCOMES = frozenset(
    {OUTCOME_PLAYER_WIN, OUTCOME_PLAYER_LOSS, OUTCOME_TIE, OUTCOME_NO_RESULT}
)

RESOLUTION_SHOWDOWN = "showdown_terminal"
RESOLUTION_OPERATOR = "operator_recovery"
RESOLUTION_SOURCES = frozenset({RESOLUTION_SHOWDOWN, RESOLUTION_OPERATOR})

BATTLE_ID_DOMAIN = b"TUGS-BLIND-BATTLE-v1"
RESULT_RECORD_DOMAIN = b"TUGS-BLIND-RESULT-RECORD-v1"
RECOVERY_CASE_DOMAIN = b"TUGS-BLIND-RESULT-RECOVERY-v1"

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SHOWDOWN_ID = re.compile(r"^[a-z0-9]{1,18}$")
_FORMAT_ID = re.compile(r"^[a-z0-9]{1,64}$")
_ROOM_ID = re.compile(r"^battle-(?P<format_id>[a-z0-9]{1,64})-(?P<number>[1-9][0-9]*)$")
_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_ZERO_HASH = "0" * 64
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "completed_results", "pending_result"})
_IDENTITY_FIELDS = frozenset(
    {
        "battle_id",
        "player_id",
        "bot_id",
        "team_id",
        "reservation_id",
        "room_id",
        "format_id",
        "registry_fingerprint",
        "correlated_at",
    }
)
_PENDING_FIELDS = frozenset({"record_version", "phase", *_IDENTITY_FIELDS})
_COMPLETED_FIELDS = frozenset(
    {
        "record_version",
        "sequence",
        "resolved_at",
        "outcome",
        "resolution_source",
        "previous_record_hash",
        "record_hash",
        *_IDENTITY_FIELDS,
    }
)


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError


def _validate_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if expected - value.keys():
        _fail(
            "result_missing_required_field",
            "{} is missing required fields".format(context),
        )
    if value.keys() - expected:
        _fail(
            "result_unexpected_field",
            "{} contains unexpected fields".format(context),
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        _fail(
            "result_path_unavailable",
            "Blind Ladder result path identity is unavailable",
        )
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _validate_existing_regular_file(path: Path, *, context: str) -> None:
    if _is_link_or_reparse(path):
        _fail(
            "result_path_unsafe",
            "{} must not be a link or reparse point".format(context),
        )
    try:
        info = os.lstat(path)
    except OSError:
        _fail(
            "result_path_unavailable",
            "{} is unavailable".format(context),
        )
    if not stat.S_ISREG(info.st_mode):
        _fail(
            "result_path_invalid",
            "{} must be a regular file".format(context),
        )
    if getattr(info, "st_nlink", 1) != 1:
        _fail(
            "result_path_unsafe",
            "{} must not be hard linked".format(context),
        )


@dataclass(frozen=True, slots=True, repr=False)
class BlindResultLedgerConfig:
    """Resolved private result path and its collision-validation authority."""

    ledger_path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    registry_path: Path = field(repr=False)
    selection_state_path: Path = field(repr=False)
    repository_root: Path = field(repr=False)

    @property
    def lock_path(self) -> Path:
        return self.ledger_path.with_name(self.ledger_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindResultLedgerConfig(configured=True)"


def validate_result_ledger_config(
    ledger_path: str | Path,
    *,
    private_root: str | Path,
    registry_path: str | Path,
    selection_state_path: str | Path,
    repository_root: str | Path | None = None,
) -> BlindResultLedgerConfig:
    """Resolve one explicit external ledger without following unsafe leaves."""

    try:
        candidate = Path(ledger_path)
        configured_private_root = Path(private_root)
        configured_registry = Path(registry_path)
        configured_state = Path(selection_state_path)
    except (TypeError, ValueError):
        _fail(
            "result_config_invalid",
            "Blind Ladder result configuration is invalid",
        )
    if not all(
        path.is_absolute()
        for path in (
            candidate,
            configured_private_root,
            configured_registry,
            configured_state,
        )
    ):
        _fail(
            "result_path_not_absolute",
            "Blind Ladder result path must be explicit and absolute",
        )
    if not candidate.name or candidate.name in {".", ".."}:
        _fail(
            "result_path_invalid",
            "Blind Ladder result path is invalid",
        )

    try:
        resolved_repository = Path(repository_root or _repository_root()).resolve(
            strict=True
        )
        resolved_private_root = configured_private_root.resolve(strict=True)
        resolved_registry = configured_registry.resolve(strict=True)
        resolved_state = configured_state.resolve(strict=False)
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail(
            "result_path_unavailable",
            "Blind Ladder result path is unavailable",
        )
    raw_ancestor = candidate.parent
    while raw_ancestor != raw_ancestor.parent:
        if raw_ancestor.exists() and _is_link_or_reparse(raw_ancestor):
            _fail(
                "result_parent_unsafe",
                "Blind Ladder result parent must be a stable directory",
            )
        raw_ancestor = raw_ancestor.parent
    if not parent.is_dir() or _is_link_or_reparse(candidate.parent):
        _fail(
            "result_parent_unsafe",
            "Blind Ladder result parent must be a stable directory",
        )
    resolved_ledger = parent / candidate.name
    if candidate.exists() or candidate.is_symlink():
        _validate_existing_regular_file(candidate, context="Blind Ladder result ledger")
        try:
            resolved_ledger = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            _fail(
                "result_path_unavailable",
                "Blind Ladder result ledger is unavailable",
            )

    resolved_lock = parent / (candidate.name + ".lock")
    raw_lock = candidate.with_name(candidate.name + ".lock")
    if raw_lock.exists() or raw_lock.is_symlink():
        _validate_existing_regular_file(raw_lock, context="Blind Ladder result lock")
        try:
            resolved_lock = raw_lock.resolve(strict=True)
        except (OSError, RuntimeError):
            _fail(
                "result_lock_path_invalid",
                "Blind Ladder result lock path is invalid",
            )

    if _is_within(resolved_ledger, resolved_repository):
        _fail(
            "result_path_not_external",
            "Blind Ladder result ledger must be outside the repository",
        )
    if _is_within(resolved_ledger, resolved_private_root):
        _fail(
            "result_path_in_deployment",
            "Blind Ladder result ledger must be outside sealed deployments",
        )
    if _is_within(resolved_lock, resolved_repository) or _is_within(
        resolved_lock, resolved_private_root
    ):
        _fail(
            "result_lock_path_unsafe",
            "Blind Ladder result lock path is unsafe",
        )

    collision_paths = {
        resolved_registry,
        resolved_state,
        resolved_state.with_name(resolved_state.name + ".lock"),
        resolved_state.with_name(resolved_state.name + ".owner.lock"),
    }
    if resolved_ledger in collision_paths or resolved_lock in collision_paths:
        _fail(
            "result_path_collision",
            "Blind Ladder result persistence collides with deployment state",
        )
    if resolved_ledger == resolved_lock:
        _fail(
            "result_path_collision",
            "Blind Ladder result ledger and lock must use different files",
        )
    return BlindResultLedgerConfig(
        resolved_ledger,
        resolved_private_root,
        resolved_registry,
        resolved_state,
        resolved_repository,
    )


def _framed_digest(
    domain: bytes,
    fields: tuple[tuple[str, str], ...],
) -> str:
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


def derive_blind_battle_id(
    *,
    reservation_id: str,
    room_id: str,
    player_id: str,
    bot_id: str,
    team_id: str,
    registry_fingerprint: str,
) -> str:
    """Derive one private battle ID without challenge-token material."""

    values = {
        "reservation_id": reservation_id,
        "room_id": room_id,
        "player_id": player_id,
        "bot_id": bot_id,
        "team_id": team_id,
        "registry_fingerprint": registry_fingerprint,
    }
    if (
        _HEX_32.fullmatch(reservation_id) is None
        or _ROOM_ID.fullmatch(room_id) is None
        or _SHOWDOWN_ID.fullmatch(player_id) is None
        or _SHOWDOWN_ID.fullmatch(bot_id) is None
        or player_id == bot_id
        or not is_valid_opaque_team_id(team_id)
        or _HEX_64.fullmatch(registry_fingerprint) is None
    ):
        _fail(
            "result_identity_invalid",
            "Blind Ladder battle identity is invalid",
        )
    return _framed_digest(
        BATTLE_ID_DOMAIN,
        tuple((label, values[label]) for label in sorted(values)),
    )


def _parse_timestamp(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        _fail(
            "result_timestamp_invalid",
            "{} timestamp is invalid".format(context),
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        _fail(
            "result_timestamp_invalid",
            "{} timestamp is invalid".format(context),
        )
    return parsed


def _format_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(
            "result_clock_invalid",
            "Blind Ladder result clock must return an aware datetime",
        )
    try:
        utc = value.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        _fail(
            "result_clock_invalid",
            "Blind Ladder result clock returned an invalid datetime",
        )
    return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True, slots=True, repr=False)
class BlindPendingBattleResult:
    record_version: int
    phase: str
    battle_id: str = field(repr=False)
    player_id: str = field(repr=False)
    bot_id: str = field(repr=False)
    team_id: str = field(repr=False)
    reservation_id: str = field(repr=False)
    room_id: str = field(repr=False)
    format_id: str
    registry_fingerprint: str = field(repr=False)
    correlated_at: str

    def __repr__(self) -> str:
        return "BlindPendingBattleResult(phase={!r})".format(self.phase)


@dataclass(frozen=True, slots=True, repr=False)
class BlindCompletedBattleResult:
    record_version: int
    sequence: int
    battle_id: str = field(repr=False)
    player_id: str = field(repr=False)
    bot_id: str = field(repr=False)
    team_id: str = field(repr=False)
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

    def __repr__(self) -> str:
        return "BlindCompletedBattleResult(sequence={!r}, outcome={!r})".format(
            self.sequence,
            self.outcome,
        )


@dataclass(frozen=True, slots=True, repr=False)
class BlindResultLedgerState:
    schema_version: int
    completed_results: tuple[BlindCompletedBattleResult, ...] = field(repr=False)
    pending_result: BlindPendingBattleResult | None = field(repr=False)

    @property
    def completed_count(self) -> int:
        return len(self.completed_results)

    @property
    def pending_count(self) -> int:
        return int(self.pending_result is not None)

    def __repr__(self) -> str:
        return (
            "BlindResultLedgerState(completed_count={!r}, pending_count={!r})"
        ).format(self.completed_count, self.pending_count)


def _pending_document(pending: BlindPendingBattleResult) -> dict[str, Any]:
    return {
        "record_version": pending.record_version,
        "phase": pending.phase,
        "battle_id": pending.battle_id,
        "player_id": pending.player_id,
        "bot_id": pending.bot_id,
        "team_id": pending.team_id,
        "reservation_id": pending.reservation_id,
        "room_id": pending.room_id,
        "format_id": pending.format_id,
        "registry_fingerprint": pending.registry_fingerprint,
        "correlated_at": pending.correlated_at,
    }


def _completed_document(record: BlindCompletedBattleResult) -> dict[str, Any]:
    return {
        "record_version": record.record_version,
        "sequence": record.sequence,
        "battle_id": record.battle_id,
        "player_id": record.player_id,
        "bot_id": record.bot_id,
        "team_id": record.team_id,
        "reservation_id": record.reservation_id,
        "room_id": record.room_id,
        "format_id": record.format_id,
        "registry_fingerprint": record.registry_fingerprint,
        "correlated_at": record.correlated_at,
        "resolved_at": record.resolved_at,
        "outcome": record.outcome,
        "resolution_source": record.resolution_source,
        "previous_record_hash": record.previous_record_hash,
        "record_hash": record.record_hash,
    }


def _state_document(state: BlindResultLedgerState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "completed_results": [
            _completed_document(record) for record in state.completed_results
        ],
        "pending_result": (
            None
            if state.pending_result is None
            else _pending_document(state.pending_result)
        ),
    }


def _validate_identity(value: dict[str, Any], *, context: str) -> None:
    if (
        not isinstance(value["battle_id"], str)
        or _HEX_64.fullmatch(value["battle_id"]) is None
        or not isinstance(value["player_id"], str)
        or _SHOWDOWN_ID.fullmatch(value["player_id"]) is None
        or not isinstance(value["bot_id"], str)
        or _SHOWDOWN_ID.fullmatch(value["bot_id"]) is None
        or value["player_id"] == value["bot_id"]
        or not is_valid_opaque_team_id(value["team_id"])
        or not isinstance(value["reservation_id"], str)
        or _HEX_32.fullmatch(value["reservation_id"]) is None
        or not isinstance(value["room_id"], str)
        or _ROOM_ID.fullmatch(value["room_id"]) is None
        or not isinstance(value["format_id"], str)
        or _FORMAT_ID.fullmatch(value["format_id"]) is None
        or not isinstance(value["registry_fingerprint"], str)
        or _HEX_64.fullmatch(value["registry_fingerprint"]) is None
    ):
        _fail(
            "result_identity_invalid",
            "{} identity is invalid".format(context),
        )
    room_match = _ROOM_ID.fullmatch(value["room_id"])
    assert room_match is not None
    if room_match.group("format_id") != value["format_id"]:
        _fail(
            "result_format_mismatch",
            "{} room format is inconsistent".format(context),
        )
    expected_battle_id = derive_blind_battle_id(
        reservation_id=value["reservation_id"],
        room_id=value["room_id"],
        player_id=value["player_id"],
        bot_id=value["bot_id"],
        team_id=value["team_id"],
        registry_fingerprint=value["registry_fingerprint"],
    )
    if value["battle_id"] != expected_battle_id:
        _fail(
            "result_battle_id_mismatch",
            "{} battle identity is inconsistent".format(context),
        )
    _parse_timestamp(value["correlated_at"], context=context)


def _validate_pending(value: Any) -> BlindPendingBattleResult | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(
            "result_pending_invalid",
            "Blind Ladder pending result must be an object",
        )
    _validate_exact_fields(value, _PENDING_FIELDS, context="Pending result")
    if (
        value["record_version"] != RESULT_RECORD_VERSION
        or type(value["record_version"]) is not int
    ):
        _fail(
            "result_record_version_unsupported",
            "Pending result record version is unsupported",
        )
    if not isinstance(value["phase"], str) or value["phase"] not in PENDING_PHASES:
        _fail(
            "result_pending_phase_invalid",
            "Pending result phase is unsupported",
        )
    _validate_identity(value, context="Pending result")
    return BlindPendingBattleResult(**value)


def _record_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("record_hash", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _framed_digest(RESULT_RECORD_DOMAIN, (("record", canonical),))


def _validate_completed(
    value: Any,
    *,
    sequence: int,
    previous_hash: str,
) -> BlindCompletedBattleResult:
    if not isinstance(value, dict):
        _fail(
            "result_completed_invalid",
            "Completed result must be an object",
        )
    _validate_exact_fields(value, _COMPLETED_FIELDS, context="Completed result")
    if (
        value["record_version"] != RESULT_RECORD_VERSION
        or type(value["record_version"]) is not int
    ):
        _fail(
            "result_record_version_unsupported",
            "Completed result record version is unsupported",
        )
    if type(value["sequence"]) is not int or value["sequence"] != sequence:
        _fail(
            "result_sequence_invalid",
            "Completed result sequence is inconsistent",
        )
    _validate_identity(value, context="Completed result")
    resolved = _parse_timestamp(value["resolved_at"], context="Completed result")
    correlated = _parse_timestamp(value["correlated_at"], context="Completed result")
    if resolved < correlated:
        _fail(
            "result_timestamp_order_invalid",
            "Completed result timestamps are inconsistent",
        )
    if not isinstance(value["outcome"], str) or value["outcome"] not in RESULT_OUTCOMES:
        _fail("result_outcome_invalid", "Completed result outcome is unsupported")
    if (
        not isinstance(value["resolution_source"], str)
        or value["resolution_source"] not in RESOLUTION_SOURCES
    ):
        _fail(
            "result_resolution_source_invalid",
            "Completed result resolution source is unsupported",
        )
    if (
        value["outcome"] == OUTCOME_NO_RESULT
        and value["resolution_source"] != RESOLUTION_OPERATOR
    ):
        _fail(
            "result_disposition_invalid",
            "No-result disposition requires explicit operator recovery",
        )
    if value["previous_record_hash"] != previous_hash:
        _fail(
            "result_history_chain_invalid",
            "Completed result history chain is inconsistent",
        )
    if (
        not isinstance(value["record_hash"], str)
        or _HEX_64.fullmatch(value["record_hash"]) is None
        or value["record_hash"] != _record_hash(value)
    ):
        _fail(
            "result_record_hash_invalid",
            "Completed result integrity hash is invalid",
        )
    return BlindCompletedBattleResult(**value)


def validate_result_ledger_document(document: Any) -> BlindResultLedgerState:
    """Validate one decoded ledger without binding history to a current pool."""

    if not isinstance(document, dict):
        _fail(
            "result_document_invalid",
            "Blind Ladder result ledger must be an object",
        )
    _validate_exact_fields(document, _TOP_LEVEL_FIELDS, context="Result ledger")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != RESULT_LEDGER_SCHEMA_VERSION
    ):
        _fail(
            "result_schema_unsupported",
            "Blind Ladder result ledger schema is unsupported",
        )
    raw_completed = document["completed_results"]
    if not isinstance(raw_completed, list):
        _fail(
            "result_completed_invalid",
            "Completed result history must be a list",
        )
    completed: list[BlindCompletedBattleResult] = []
    previous_hash = _ZERO_HASH
    battle_ids: set[str] = set()
    room_ids: set[str] = set()
    reservation_ids: set[str] = set()
    for sequence, value in enumerate(raw_completed, start=1):
        record = _validate_completed(
            value,
            sequence=sequence,
            previous_hash=previous_hash,
        )
        if (
            record.battle_id in battle_ids
            or record.room_id in room_ids
            or record.reservation_id in reservation_ids
        ):
            _fail(
                "result_history_identity_reused",
                "Completed result history reuses a private identity",
            )
        battle_ids.add(record.battle_id)
        room_ids.add(record.room_id)
        reservation_ids.add(record.reservation_id)
        previous_hash = record.record_hash
        completed.append(record)

    pending = _validate_pending(document["pending_result"])
    if pending is not None and (
        pending.battle_id in battle_ids
        or pending.room_id in room_ids
        or pending.reservation_id in reservation_ids
    ):
        _fail(
            "result_pending_identity_reused",
            "Pending result reuses completed identity",
        )
    return BlindResultLedgerState(
        RESULT_LEDGER_SCHEMA_VERSION,
        tuple(completed),
        pending,
    )


def _decode_ledger(raw: bytes) -> BlindResultLedgerState:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(
            "result_encoding_invalid",
            "Blind Ladder result ledger must not contain a UTF-8 BOM",
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail(
            "result_encoding_invalid",
            "Blind Ladder result ledger is not valid UTF-8",
        )
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJsonFieldError:
        _fail(
            "result_duplicate_field",
            "Blind Ladder result ledger contains duplicate fields",
        )
    except (json.JSONDecodeError, ValueError):
        _fail(
            "result_json_invalid",
            "Blind Ladder result ledger contains malformed JSON",
        )
    return validate_result_ledger_document(document)


def _regular_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _opened_file_identity(info: os.stat_result) -> tuple[int, ...]:
    """Return fields comparable between path stat and an open Windows handle."""

    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _stable_read_bytes(path: Path) -> bytes:
    """Read one regular non-link file while proving its identity stayed fixed."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        _fail(
            "result_not_initialized",
            "Blind Ladder result ledger has not been initialized",
        )
    except OSError:
        _fail(
            "result_file_unreadable",
            "Blind Ladder result ledger could not be read",
        )
    _validate_existing_regular_file(path, context="Blind Ladder result ledger")
    try:
        before = os.lstat(path)
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _opened_file_identity(
                before
            ) != _opened_file_identity(opened):
                _fail(
                    "result_file_changed",
                    "Blind Ladder result ledger changed during validation",
                )
            raw = stream.read(before.st_size + 1)
            if len(raw) != before.st_size:
                _fail(
                    "result_file_changed",
                    "Blind Ladder result ledger changed during validation",
                )
        after = os.lstat(path)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail(
            "result_file_unreadable",
            "Blind Ladder result ledger could not be read",
        )
    if _is_link_or_reparse(path) or _regular_file_identity(
        before
    ) != _regular_file_identity(after):
        _fail(
            "result_file_changed",
            "Blind Ladder result ledger changed during validation",
        )
    return raw


def _serialize_ledger(state: BlindResultLedgerState) -> bytes:
    validated = validate_result_ledger_document(_state_document(state))
    return (
        json.dumps(
            _state_document(validated),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def _write_atomic(
    config: BlindResultLedgerConfig,
    state: BlindResultLedgerState,
    *,
    replace_existing: bool,
) -> None:
    payload = _serialize_ledger(state)
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".{}-".format(config.ledger_path.name),
            suffix=".tmp",
            dir=config.ledger_path.parent,
        )
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if replace_existing:
            os.replace(temporary_path, config.ledger_path)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, config.ledger_path)
            except FileExistsError:
                _fail(
                    "result_target_exists",
                    "Blind Ladder result ledger already exists",
                )
            temporary_path.unlink()
            temporary_path = None
        _fsync_directory(config.ledger_path.parent)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail(
            "result_atomic_write_failed",
            "Blind Ladder result ledger could not be persisted",
        )
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _identity_tuple(
    pending: BlindPendingBattleResult,
) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        pending.battle_id,
        pending.player_id,
        pending.bot_id,
        pending.team_id,
        pending.reservation_id,
        pending.room_id,
        pending.format_id,
        pending.registry_fingerprint,
    )


class BlindResultLedgerStore:
    """One-lock transactional API for append-only result history."""

    def __init__(
        self,
        config: BlindResultLedgerConfig,
        *,
        lock_timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, BlindResultLedgerConfig):
            _fail(
                "result_config_invalid",
                "Blind Ladder result configuration is invalid",
            )
        self._config = config
        self._lock_timeout = lock_timeout_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._history_baseline: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return "BlindResultLedgerStore(configured=True)"

    def _validated_config(self) -> BlindResultLedgerConfig:
        return validate_result_ledger_config(
            self._config.ledger_path,
            private_root=self._config.private_root,
            registry_path=self._config.registry_path,
            selection_state_path=self._config.selection_state_path,
            repository_root=self._config.repository_root,
        )

    def _remember_history(self, state: BlindResultLedgerState) -> None:
        observed = tuple(record.record_hash for record in state.completed_results)
        if observed[: len(self._history_baseline)] != self._history_baseline:
            _fail(
                "result_history_changed",
                "Blind Ladder completed result history changed",
            )
        self._history_baseline = observed

    def _load_unlocked(
        self,
        config: BlindResultLedgerConfig,
    ) -> BlindResultLedgerState:
        try:
            raw = _stable_read_bytes(config.ledger_path)
        except FileNotFoundError:
            _fail(
                "result_not_initialized",
                "Blind Ladder result ledger has not been initialized",
            )
        state = _decode_ledger(raw)
        self._remember_history(state)
        return state

    def initialize_empty(self) -> BlindResultLedgerState:
        config = self._validated_config()
        state = BlindResultLedgerState(RESULT_LEDGER_SCHEMA_VERSION, (), None)
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            if config.ledger_path.exists() or config.ledger_path.is_symlink():
                existing = self._load_unlocked(config)
                if existing.completed_count or existing.pending_count:
                    _fail(
                        "result_ledger_not_empty",
                        "Blind Ladder result ledger already contains history",
                    )
                return existing
            _write_atomic(config, state, replace_existing=False)
            self._remember_history(state)
        return state

    def load(self) -> BlindResultLedgerState:
        config = self._validated_config()
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            return self._load_unlocked(config)

    def require_ready(self) -> BlindResultLedgerState:
        state = self.load()
        if state.pending_result is not None:
            _fail(
                "result_recovery_required",
                "Blind Ladder result ledger requires explicit recovery",
            )
        return state

    def create_pending_intent(
        self,
        *,
        player_id: str,
        bot_id: str,
        team_id: str,
        reservation_id: str,
        room_id: str,
        format_id: str,
        registry_fingerprint: str,
    ) -> BlindPendingBattleResult:
        battle_id = derive_blind_battle_id(
            reservation_id=reservation_id,
            room_id=room_id,
            player_id=player_id,
            bot_id=bot_id,
            team_id=team_id,
            registry_fingerprint=registry_fingerprint,
        )
        candidate = BlindPendingBattleResult(
            RESULT_RECORD_VERSION,
            PENDING_ROOM_VALIDATED,
            battle_id,
            player_id,
            bot_id,
            team_id,
            reservation_id,
            room_id,
            format_id,
            registry_fingerprint,
            _format_timestamp(self._clock()),
        )
        validate_result_ledger_document(
            {
                "schema_version": RESULT_LEDGER_SCHEMA_VERSION,
                "completed_results": [],
                "pending_result": _pending_document(candidate),
            }
        )
        config = self._validated_config()
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            state = self._load_unlocked(config)
            for record in state.completed_results:
                if record.battle_id == battle_id:
                    _fail(
                        "result_battle_completed",
                        "Completed battle cannot return to pending",
                    )
                if record.room_id == room_id:
                    _fail(
                        "result_room_identity_conflict",
                        "Battle room identity is already completed",
                    )
                if record.reservation_id == reservation_id:
                    _fail(
                        "result_reservation_identity_conflict",
                        "Reservation identity is already completed",
                    )
            existing = state.pending_result
            if existing is not None:
                if _identity_tuple(existing) == _identity_tuple(candidate):
                    return existing
                if existing.battle_id == battle_id:
                    code = "result_battle_identity_conflict"
                elif existing.room_id == room_id:
                    code = "result_room_identity_conflict"
                elif existing.reservation_id == reservation_id:
                    code = "result_reservation_identity_conflict"
                else:
                    code = "result_pending_exists"
                _fail(code, "Blind Ladder result intent conflicts with pending state")
            updated = BlindResultLedgerState(
                RESULT_LEDGER_SCHEMA_VERSION,
                state.completed_results,
                candidate,
            )
            _write_atomic(config, updated, replace_existing=True)
            return candidate

    def mark_selection_committed(self, battle_id: str) -> BlindPendingBattleResult:
        config = self._validated_config()
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            state = self._load_unlocked(config)
            pending = state.pending_result
            if pending is None or pending.battle_id != battle_id:
                _fail(
                    "result_pending_mismatch",
                    "Blind Ladder pending result identity does not match",
                )
            if pending.phase == PENDING_AWAITING_TERMINAL:
                return pending
            if pending.phase != PENDING_ROOM_VALIDATED:
                _fail(
                    "result_pending_phase_invalid",
                    "Blind Ladder pending result phase is invalid",
                )
            updated_pending = BlindPendingBattleResult(
                pending.record_version,
                PENDING_AWAITING_TERMINAL,
                pending.battle_id,
                pending.player_id,
                pending.bot_id,
                pending.team_id,
                pending.reservation_id,
                pending.room_id,
                pending.format_id,
                pending.registry_fingerprint,
                pending.correlated_at,
            )
            updated = BlindResultLedgerState(
                RESULT_LEDGER_SCHEMA_VERSION,
                state.completed_results,
                updated_pending,
            )
            _write_atomic(config, updated, replace_existing=True)
            return updated_pending

    def _completed_for(
        self,
        state: BlindResultLedgerState,
        battle_id: str,
    ) -> BlindCompletedBattleResult | None:
        return next(
            (
                record
                for record in state.completed_results
                if record.battle_id == battle_id
            ),
            None,
        )

    def finalize_terminal(
        self,
        battle_id: str,
        outcome: str,
    ) -> BlindCompletedBattleResult:
        if not isinstance(outcome, str) or outcome not in {
            OUTCOME_PLAYER_WIN,
            OUTCOME_PLAYER_LOSS,
            OUTCOME_TIE,
        }:
            _fail(
                "result_terminal_outcome_invalid",
                "Automatic terminal result outcome is invalid",
            )
        return self._finalize(
            battle_id,
            outcome,
            resolution_source=RESOLUTION_SHOWDOWN,
        )

    def _finalize(
        self,
        battle_id: str,
        outcome: str,
        *,
        resolution_source: str,
    ) -> BlindCompletedBattleResult:
        if not isinstance(battle_id, str) or _HEX_64.fullmatch(battle_id) is None:
            _fail(
                "result_battle_id_invalid",
                "Blind Ladder battle ID is invalid",
            )
        config = self._validated_config()
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            state = self._load_unlocked(config)
            completed = self._completed_for(state, battle_id)
            if completed is not None:
                if (
                    completed.outcome == outcome
                    and completed.resolution_source == resolution_source
                ):
                    return completed
                _fail(
                    "result_outcome_conflict",
                    "Completed battle result is immutable",
                )
            pending = state.pending_result
            if pending is None or pending.battle_id != battle_id:
                _fail(
                    "result_pending_mismatch",
                    "Blind Ladder pending result identity does not match",
                )
            if (
                resolution_source == RESOLUTION_SHOWDOWN
                and pending.phase != PENDING_AWAITING_TERMINAL
            ):
                _fail(
                    "result_pending_phase_invalid",
                    "Automatic result requires selection-committed state",
                )
            previous_hash = (
                state.completed_results[-1].record_hash
                if state.completed_results
                else _ZERO_HASH
            )
            document = {
                "record_version": RESULT_RECORD_VERSION,
                "sequence": state.completed_count + 1,
                "battle_id": pending.battle_id,
                "player_id": pending.player_id,
                "bot_id": pending.bot_id,
                "team_id": pending.team_id,
                "reservation_id": pending.reservation_id,
                "room_id": pending.room_id,
                "format_id": pending.format_id,
                "registry_fingerprint": pending.registry_fingerprint,
                "correlated_at": pending.correlated_at,
                "resolved_at": _format_timestamp(self._clock()),
                "outcome": outcome,
                "resolution_source": resolution_source,
                "previous_record_hash": previous_hash,
                "record_hash": "",
            }
            document["record_hash"] = _record_hash(document)
            record = _validate_completed(
                document,
                sequence=state.completed_count + 1,
                previous_hash=previous_hash,
            )
            updated = BlindResultLedgerState(
                RESULT_LEDGER_SCHEMA_VERSION,
                (*state.completed_results, record),
                None,
            )
            _write_atomic(config, updated, replace_existing=True)
            self._remember_history(updated)
            return record

    def recovery_case(self) -> str:
        state = self.load()
        return self._recovery_case_for(state)

    @staticmethod
    def _recovery_case_for(state: BlindResultLedgerState) -> str:
        pending = state.pending_result
        if pending is None:
            _fail(
                "result_recovery_not_applicable",
                "Blind Ladder result recovery is not applicable",
            )
        history_hash = (
            state.completed_results[-1].record_hash
            if state.completed_results
            else _ZERO_HASH
        )
        canonical_pending = json.dumps(
            _pending_document(pending),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return _framed_digest(
            RECOVERY_CASE_DOMAIN,
            (("history_hash", history_hash), ("pending", canonical_pending)),
        )

    def resolve_pending(
        self,
        recovery_case: str,
        outcome: str,
    ) -> BlindCompletedBattleResult:
        if not isinstance(outcome, str) or outcome not in RESULT_OUTCOMES:
            _fail(
                "result_outcome_invalid",
                "Blind Ladder recovery outcome is invalid",
            )
        if (
            not isinstance(recovery_case, str)
            or _HEX_64.fullmatch(recovery_case) is None
        ):
            _fail(
                "result_recovery_case_invalid",
                "Blind Ladder recovery case is invalid",
            )
        config = self._validated_config()
        with BlindPoolStateLock(
            config.lock_path,
            timeout_seconds=self._lock_timeout,
        ):
            state = self._load_unlocked(config)
            expected = self._recovery_case_for(state)
            if not hmac.compare_digest(recovery_case, expected):
                _fail(
                    "result_recovery_case_mismatch",
                    "Blind Ladder recovery case is stale or mismatched",
                )
            pending = state.pending_result
            assert pending is not None
            previous_hash = (
                state.completed_results[-1].record_hash
                if state.completed_results
                else _ZERO_HASH
            )
            document = {
                "record_version": RESULT_RECORD_VERSION,
                "sequence": state.completed_count + 1,
                "battle_id": pending.battle_id,
                "player_id": pending.player_id,
                "bot_id": pending.bot_id,
                "team_id": pending.team_id,
                "reservation_id": pending.reservation_id,
                "room_id": pending.room_id,
                "format_id": pending.format_id,
                "registry_fingerprint": pending.registry_fingerprint,
                "correlated_at": pending.correlated_at,
                "resolved_at": _format_timestamp(self._clock()),
                "outcome": outcome,
                "resolution_source": RESOLUTION_OPERATOR,
                "previous_record_hash": previous_hash,
                "record_hash": "",
            }
            document["record_hash"] = _record_hash(document)
            record = _validate_completed(
                document,
                sequence=state.completed_count + 1,
                previous_hash=previous_hash,
            )
            updated = BlindResultLedgerState(
                RESULT_LEDGER_SCHEMA_VERSION,
                (*state.completed_results, record),
                None,
            )
            _write_atomic(config, updated, replace_existing=True)
            self._remember_history(updated)
            return record
