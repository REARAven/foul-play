"""Strict persistence for the derived Blind Ladder Elo-v1 projection."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, NoReturn

from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .rating import (
    INITIAL_RATING,
    K_FACTOR,
    RATING_ALGORITHM_ID,
    RATING_SCALE,
    RATING_STATE_SCHEMA_VERSION,
    ROUNDING_POLICY,
    BlindOpponentRating,
    BlindPlayerRating,
    BlindRatingState,
    BlindRatingUpdate,
    apply_completed_result,
    derive_rating_state,
    validate_rating_state,
)
from .result_ledger import BlindResultLedgerState


RATING_STATUS_SYNCED = "synced"
RATING_STATUS_BEHIND = "behind"

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "processed_sequence",
        "processed_record_hash",
        "rated_results",
        "players",
        "opponents",
    }
)
_ALGORITHM_FIELDS = frozenset(
    {"id", "initial_rating", "k_factor", "scale", "rounding_policy"}
)
_PLAYER_FIELDS = frozenset(
    {
        "rating",
        "games_played",
        "wins",
        "losses",
        "ties",
        "peak_rating",
        "streak_kind",
        "streak_length",
    }
)
_OPPONENT_FIELDS = frozenset({"rating", "games_played"})
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        _fail("rating_path_unavailable", "Blind Ladder rating path is unavailable")
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _validate_existing_regular_file(path: Path, *, context: str) -> None:
    if _is_link_or_reparse(path):
        _fail("rating_path_unsafe", "{} must not be a link".format(context))
    try:
        info = os.lstat(path)
    except OSError:
        _fail("rating_path_unavailable", "{} is unavailable".format(context))
    if not stat.S_ISREG(info.st_mode):
        _fail("rating_path_invalid", "{} must be a regular file".format(context))
    if getattr(info, "st_nlink", 1) != 1:
        _fail("rating_path_unsafe", "{} must not be hard linked".format(context))


@dataclass(frozen=True, slots=True, repr=False)
class BlindRatingStateConfig:
    rating_path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    registry_path: Path = field(repr=False)
    selection_state_path: Path = field(repr=False)
    result_ledger_path: Path = field(repr=False)
    repository_root: Path = field(repr=False)

    @property
    def lock_path(self) -> Path:
        return self.rating_path.with_name(self.rating_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindRatingStateConfig(configured=True)"


def validate_rating_state_config(
    rating_path: str | Path,
    *,
    private_root: str | Path,
    registry_path: str | Path,
    selection_state_path: str | Path,
    result_ledger_path: str | Path,
    repository_root: str | Path | None = None,
) -> BlindRatingStateConfig:
    """Resolve one explicit external rating snapshot and reject collisions."""

    try:
        candidate = Path(rating_path)
        configured_private_root = Path(private_root)
        configured_registry = Path(registry_path)
        configured_state = Path(selection_state_path)
        configured_result = Path(result_ledger_path)
    except (TypeError, ValueError):
        _fail("rating_config_invalid", "Blind Ladder rating configuration is invalid")
    paths = (
        candidate,
        configured_private_root,
        configured_registry,
        configured_state,
        configured_result,
    )
    if not all(path.is_absolute() for path in paths):
        _fail(
            "rating_path_not_absolute",
            "Blind Ladder rating path must be explicit and absolute",
        )
    if not candidate.name or candidate.name in {".", ".."}:
        _fail("rating_path_invalid", "Blind Ladder rating path is invalid")
    try:
        resolved_repository = Path(repository_root or _repository_root()).resolve(
            strict=True
        )
        resolved_private_root = configured_private_root.resolve(strict=True)
        resolved_registry = configured_registry.resolve(strict=True)
        resolved_state = configured_state.resolve(strict=False)
        resolved_result = configured_result.resolve(strict=False)
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("rating_path_unavailable", "Blind Ladder rating path is unavailable")
    raw_ancestor = candidate.parent
    while raw_ancestor != raw_ancestor.parent:
        if raw_ancestor.exists() and _is_link_or_reparse(raw_ancestor):
            _fail(
                "rating_parent_unsafe",
                "Blind Ladder rating parent must be a stable directory",
            )
        raw_ancestor = raw_ancestor.parent
    if not parent.is_dir() or _is_link_or_reparse(candidate.parent):
        _fail(
            "rating_parent_unsafe",
            "Blind Ladder rating parent must be a stable directory",
        )
    resolved_rating = parent / candidate.name
    if candidate.exists() or candidate.is_symlink():
        _validate_existing_regular_file(candidate, context="Blind Ladder rating state")
        try:
            resolved_rating = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            _fail(
                "rating_path_unavailable",
                "Blind Ladder rating state is unavailable",
            )
    raw_lock = candidate.with_name(candidate.name + ".lock")
    resolved_lock = parent / raw_lock.name
    if raw_lock.exists() or raw_lock.is_symlink():
        _validate_existing_regular_file(raw_lock, context="Blind Ladder rating lock")
        try:
            resolved_lock = raw_lock.resolve(strict=True)
        except (OSError, RuntimeError):
            _fail(
                "rating_lock_path_invalid",
                "Blind Ladder rating lock path is invalid",
            )

    if _is_within(resolved_rating, resolved_repository):
        _fail(
            "rating_path_not_external",
            "Blind Ladder rating state must be outside the repository",
        )
    if _is_within(resolved_rating, resolved_private_root):
        _fail(
            "rating_path_in_deployment",
            "Blind Ladder rating state must be outside sealed deployments",
        )
    if _is_within(resolved_lock, resolved_repository) or _is_within(
        resolved_lock, resolved_private_root
    ):
        _fail("rating_lock_path_unsafe", "Blind Ladder rating lock path is unsafe")

    collision_paths = {
        resolved_registry,
        resolved_state,
        resolved_state.with_name(resolved_state.name + ".lock"),
        resolved_state.with_name(resolved_state.name + ".owner.lock"),
        resolved_result,
        resolved_result.with_name(resolved_result.name + ".lock"),
    }
    if (
        resolved_rating in collision_paths
        or resolved_lock in collision_paths
        or resolved_rating == resolved_lock
    ):
        _fail(
            "rating_path_collision",
            "Blind Ladder rating persistence collides with existing state",
        )
    return BlindRatingStateConfig(
        resolved_rating,
        resolved_private_root,
        resolved_registry,
        resolved_state,
        resolved_result,
        resolved_repository,
    )


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonFieldError
        value[key] = item
    return value


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError


def _validate_exact_fields(
    value: Any,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("rating_document_invalid", "{} must be an object".format(context))
    if expected - value.keys():
        _fail("rating_missing_required_field", "{} is incomplete".format(context))
    if value.keys() - expected:
        _fail("rating_unexpected_field", "{} has unexpected fields".format(context))
    return value


def validate_rating_state_document(document: Any) -> BlindRatingState:
    """Validate and freeze a strict schema-1 rating-state document."""

    root = _validate_exact_fields(document, _TOP_LEVEL_FIELDS, context="Rating state")
    algorithm = _validate_exact_fields(
        root["algorithm"], _ALGORITHM_FIELDS, context="Rating algorithm"
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != RATING_STATE_SCHEMA_VERSION
    ):
        _fail("rating_schema_unsupported", "Rating state schema is unsupported")
    if (
        algorithm["id"] != RATING_ALGORITHM_ID
        or type(algorithm["initial_rating"]) is not int
        or algorithm["initial_rating"] != INITIAL_RATING
        or type(algorithm["k_factor"]) is not int
        or algorithm["k_factor"] != K_FACTOR
        or type(algorithm["scale"]) is not int
        or algorithm["scale"] != RATING_SCALE
        or algorithm["rounding_policy"] != ROUNDING_POLICY
    ):
        _fail("rating_algorithm_unsupported", "Rating algorithm is unsupported")
    if not isinstance(root["players"], dict) or not isinstance(root["opponents"], dict):
        _fail("rating_mapping_invalid", "Rating mappings are invalid")

    players: list[BlindPlayerRating] = []
    for player_id, value in root["players"].items():
        entry = _validate_exact_fields(value, _PLAYER_FIELDS, context="Player rating")
        players.append(
            BlindPlayerRating(
                player_id,
                entry["rating"],
                entry["games_played"],
                entry["wins"],
                entry["losses"],
                entry["ties"],
                entry["peak_rating"],
                entry["streak_kind"],
                entry["streak_length"],
            )
        )
    opponents: list[BlindOpponentRating] = []
    for team_id, value in root["opponents"].items():
        entry = _validate_exact_fields(
            value, _OPPONENT_FIELDS, context="Opponent rating"
        )
        opponents.append(
            BlindOpponentRating(team_id, entry["rating"], entry["games_played"])
        )
    state = BlindRatingState(
        root["schema_version"],
        algorithm["id"],
        algorithm["initial_rating"],
        algorithm["k_factor"],
        algorithm["scale"],
        algorithm["rounding_policy"],
        root["processed_sequence"],
        root["processed_record_hash"],
        root["rated_results"],
        tuple(players),
        tuple(opponents),
    )
    return validate_rating_state(state)


def _state_document(state: BlindRatingState) -> dict[str, Any]:
    validate_rating_state(state)
    return {
        "schema_version": state.schema_version,
        "algorithm": {
            "id": state.algorithm_id,
            "initial_rating": state.initial_rating,
            "k_factor": state.k_factor,
            "scale": state.scale,
            "rounding_policy": state.rounding_policy,
        },
        "processed_sequence": state.processed_sequence,
        "processed_record_hash": state.processed_record_hash,
        "rated_results": state.rated_results,
        "players": {
            player.player_id: {
                "rating": player.rating,
                "games_played": player.games_played,
                "wins": player.wins,
                "losses": player.losses,
                "ties": player.ties,
                "peak_rating": player.peak_rating,
                "streak_kind": player.streak_kind,
                "streak_length": player.streak_length,
            }
            for player in state.players
        },
        "opponents": {
            opponent.team_id: {
                "rating": opponent.rating,
                "games_played": opponent.games_played,
            }
            for opponent in state.opponents
        },
    }


def _decode_state(raw: bytes) -> BlindRatingState:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("rating_encoding_invalid", "Rating state must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("rating_encoding_invalid", "Rating state is not valid UTF-8")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJsonFieldError:
        _fail("rating_duplicate_field", "Rating state contains duplicate fields")
    except (json.JSONDecodeError, ValueError):
        _fail("rating_json_invalid", "Rating state contains malformed JSON")
    return validate_rating_state_document(document)


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
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _stable_read_bytes(path: Path) -> bytes:
    try:
        os.lstat(path)
    except FileNotFoundError:
        _fail("rating_not_initialized", "Rating state has not been initialized")
    except OSError:
        _fail("rating_file_unreadable", "Rating state could not be read")
    _validate_existing_regular_file(path, context="Blind Ladder rating state")
    try:
        before = os.lstat(path)
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _opened_file_identity(
                before
            ) != _opened_file_identity(opened):
                _fail("rating_file_changed", "Rating state changed during validation")
            raw = stream.read(before.st_size + 1)
            if len(raw) != before.st_size:
                _fail("rating_file_changed", "Rating state changed during validation")
        after = os.lstat(path)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail("rating_file_unreadable", "Rating state could not be read")
    if _is_link_or_reparse(path) or _regular_file_identity(
        before
    ) != _regular_file_identity(after):
        _fail("rating_file_changed", "Rating state changed during validation")
    return raw


def _serialize_state(state: BlindRatingState) -> bytes:
    validated = validate_rating_state_document(_state_document(state))
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
    config: BlindRatingStateConfig,
    state: BlindRatingState,
    *,
    replace_existing: bool,
) -> None:
    payload = _serialize_state(state)
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".{}-".format(config.rating_path.name),
            suffix=".tmp",
            dir=config.rating_path.parent,
        )
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if replace_existing:
            os.replace(temporary_path, config.rating_path)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, config.rating_path)
            except FileExistsError:
                _fail("rating_target_exists", "Rating state already exists")
            temporary_path.unlink()
            temporary_path = None
        _fsync_directory(config.rating_path.parent)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail("rating_atomic_write_failed", "Rating state could not be persisted")
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


def _validate_ledger(ledger: BlindResultLedgerState) -> BlindResultLedgerState:
    if (
        not isinstance(ledger, BlindResultLedgerState)
        or ledger.pending_result is not None
    ):
        _fail("rating_ledger_not_ready", "Result ledger is not ready for ratings")
    return ledger


def _projection_status(
    state: BlindRatingState,
    ledger: BlindResultLedgerState,
) -> str:
    ledger = _validate_ledger(ledger)
    validate_rating_state(state)
    count = ledger.completed_count
    if state.processed_sequence > count:
        _fail("rating_state_ahead", "Rating state is ahead of result history")
    expected_prefix = derive_rating_state(
        ledger.completed_results[: state.processed_sequence]
    )
    if state != expected_prefix:
        _fail("rating_state_diverged", "Rating state diverges from result history")
    return (
        RATING_STATUS_SYNCED
        if state.processed_sequence == count
        else RATING_STATUS_BEHIND
    )


@dataclass(frozen=True, slots=True, repr=False)
class BlindRatingSyncResult:
    state: BlindRatingState = field(repr=False)
    status_before: str
    processed_before: int
    updates: tuple[BlindRatingUpdate, ...]

    @property
    def applied_results(self) -> int:
        return self.state.processed_sequence - self.processed_before

    @property
    def last_update(self) -> BlindRatingUpdate | None:
        return self.updates[-1] if self.updates else None

    def __repr__(self) -> str:
        return (
            "BlindRatingSyncResult(status_before={!r}, processed_results={!r})".format(
                self.status_before,
                self.state.processed_results,
            )
        )


class BlindRatingStateStore:
    """Independent locked persistence for a rebuildable ledger projection."""

    def __init__(
        self,
        config: BlindRatingStateConfig,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(config, BlindRatingStateConfig):
            _fail(
                "rating_config_invalid", "Blind Ladder rating configuration is invalid"
            )
        self._config = config
        self._lock_timeout = lock_timeout_seconds

    def __repr__(self) -> str:
        return "BlindRatingStateStore(configured=True)"

    def _validated_config(self) -> BlindRatingStateConfig:
        return validate_rating_state_config(
            self._config.rating_path,
            private_root=self._config.private_root,
            registry_path=self._config.registry_path,
            selection_state_path=self._config.selection_state_path,
            result_ledger_path=self._config.result_ledger_path,
            repository_root=self._config.repository_root,
        )

    def _load_unlocked(self, config: BlindRatingStateConfig) -> BlindRatingState:
        return _decode_state(_stable_read_bytes(config.rating_path))

    def load(self) -> BlindRatingState:
        config = self._validated_config()
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            return self._load_unlocked(config)

    def initialize(self, ledger: BlindResultLedgerState) -> BlindRatingState:
        expected = derive_rating_state(_validate_ledger(ledger).completed_results)
        config = self._validated_config()
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            if config.rating_path.exists() or config.rating_path.is_symlink():
                existing = self._load_unlocked(config)
                if existing != expected:
                    _fail(
                        "rating_initialization_conflict",
                        "Existing rating state conflicts with result history",
                    )
                return existing
            _write_atomic(config, expected, replace_existing=False)
        return expected

    def status(self, ledger: BlindResultLedgerState) -> tuple[str, BlindRatingState]:
        config = self._validated_config()
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            return _projection_status(state, ledger), state

    def verify(self, ledger: BlindResultLedgerState) -> BlindRatingState:
        status, state = self.status(ledger)
        if status != RATING_STATUS_SYNCED:
            _fail("rating_state_behind", "Rating state is behind result history")
        expected = derive_rating_state(_validate_ledger(ledger).completed_results)
        if state != expected:
            _fail("rating_state_diverged", "Rating state diverges from result history")
        return state

    def sync(self, ledger: BlindResultLedgerState) -> BlindRatingSyncResult:
        ledger = _validate_ledger(ledger)
        config = self._validated_config()
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            status = _projection_status(state, ledger)
            processed_before = state.processed_sequence
            updates: list[BlindRatingUpdate] = []
            if status == RATING_STATUS_BEHIND:
                for record in ledger.completed_results[state.processed_sequence :]:
                    state, update = apply_completed_result(state, record)
                    if update is not None:
                        updates.append(update)
                expected = derive_rating_state(ledger.completed_results)
                if state != expected:
                    _fail(
                        "rating_state_diverged",
                        "Rating state diverges from result history",
                    )
                _write_atomic(config, state, replace_existing=True)
            return BlindRatingSyncResult(
                state,
                status,
                processed_before,
                tuple(updates),
            )

    def rebuild(self, ledger: BlindResultLedgerState) -> BlindRatingState:
        expected = derive_rating_state(_validate_ledger(ledger).completed_results)
        config = self._validated_config()
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            _write_atomic(
                config,
                expected,
                replace_existing=config.rating_path.exists()
                or config.rating_path.is_symlink(),
            )
        return expected
