"""Strict persistence for the team-era Blind Ladder Elo-v1 projection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, NoReturn

from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .rating import (
    INITIAL_RATING,
    K_FACTOR,
    RATING_ALGORITHM_ID,
    RATING_SCALE,
    ROUNDING_POLICY,
)
from .team_persistence import (
    TeamStateFileConfig,
    stable_read_bytes,
    validate_team_state_file_config,
    write_atomic,
)
from .team_rating import (
    TEAM_RATING_STATE_SCHEMA_VERSION,
    BlindTeamRating,
    BlindTeamRatingState,
    BlindTeamRatingUpdate,
    apply_team_battle_result,
    derive_team_rating_state,
    validate_team_rating_state,
)
from .team_result_ledger import BlindTeamResultLedgerState


TEAM_RATING_STATE_PATH_ENV = "TUGS_BLIND_TEAM_RATING_STATE"
TEAM_RATING_STATUS_SYNCED = "synced"
TEAM_RATING_STATUS_BEHIND = "behind"
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "processed_sequence",
        "processed_record_hash",
        "rated_results",
        "teams",
    }
)
_ALGORITHM_FIELDS = frozenset(
    {"id", "initial_rating", "k_factor", "scale", "rounding_policy"}
)
_TEAM_FIELDS = frozenset(
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


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamRatingStateConfig:
    rating_path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    canonical_registry_path: Path = field(repr=False)
    selection_state_path: Path = field(repr=False)
    result_ledger_path: Path = field(repr=False)
    public_registry_path: Path = field(repr=False)
    repository_root: Path | None = field(default=None, repr=False)

    @property
    def lock_path(self) -> Path:
        return self.rating_path.with_name(self.rating_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindTeamRatingStateConfig(configured=True)"


def validate_team_rating_state_config(
    rating_path: str | Path,
    *,
    private_root: str | Path,
    canonical_registry_path: str | Path,
    selection_state_path: str | Path,
    result_ledger_path: str | Path,
    public_registry_path: str | Path,
    repository_root: str | Path | None = None,
) -> BlindTeamRatingStateConfig:
    collisions = (
        Path(canonical_registry_path),
        Path(selection_state_path),
        Path(result_ledger_path),
        Path(public_registry_path),
    )
    validated = validate_team_state_file_config(
        rating_path,
        private_root=private_root,
        collision_paths=collisions,
        code_prefix="team_rating",
        repository_root=repository_root,
    )
    return BlindTeamRatingStateConfig(
        validated.path,
        validated.private_root,
        *(item.resolve(strict=False) for item in collisions),
        validated.repository_root,
    )


def _file_config(config: BlindTeamRatingStateConfig) -> TeamStateFileConfig:
    return validate_team_state_file_config(
        config.rating_path,
        private_root=config.private_root,
        collision_paths=(
            config.canonical_registry_path,
            config.selection_state_path,
            config.result_ledger_path,
            config.public_registry_path,
        ),
        code_prefix="team_rating",
        repository_root=config.repository_root,
    )


def _state_document(state: BlindTeamRatingState) -> dict[str, Any]:
    validate_team_rating_state(state)
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
        "teams": {
            team.team_id: {
                "rating": team.rating,
                "games_played": team.games_played,
                "wins": team.wins,
                "losses": team.losses,
                "ties": team.ties,
                "peak_rating": team.peak_rating,
                "streak_kind": team.streak_kind,
                "streak_length": team.streak_length,
            }
            for team in state.teams
        },
    }


def validate_team_rating_state_document(document: Any) -> BlindTeamRatingState:
    if not isinstance(document, dict) or set(document) != _TOP_FIELDS:
        _fail("team_rating_document_invalid", "Team rating document is invalid")
    algorithm = document["algorithm"]
    if not isinstance(algorithm, dict) or set(algorithm) != _ALGORITHM_FIELDS:
        _fail("team_rating_algorithm_invalid", "Team rating algorithm is invalid")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != TEAM_RATING_STATE_SCHEMA_VERSION
        or algorithm["id"] != RATING_ALGORITHM_ID
        or algorithm["initial_rating"] != INITIAL_RATING
        or algorithm["k_factor"] != K_FACTOR
        or algorithm["scale"] != RATING_SCALE
        or algorithm["rounding_policy"] != ROUNDING_POLICY
    ):
        _fail(
            "team_rating_algorithm_unsupported", "Team rating algorithm is unsupported"
        )
    raw_teams = document["teams"]
    if not isinstance(raw_teams, dict):
        _fail("team_rating_mapping_invalid", "Team rating mapping is invalid")
    teams: list[BlindTeamRating] = []
    for team_id, raw in raw_teams.items():
        if not isinstance(raw, dict) or set(raw) != _TEAM_FIELDS:
            _fail("team_rating_entry_invalid", "Team rating entry is invalid")
        teams.append(
            BlindTeamRating(
                team_id,
                raw["rating"],
                raw["games_played"],
                raw["wins"],
                raw["losses"],
                raw["ties"],
                raw["peak_rating"],
                raw["streak_kind"],
                raw["streak_length"],
            )
        )
    return validate_team_rating_state(
        BlindTeamRatingState(
            document["schema_version"],
            algorithm["id"],
            algorithm["initial_rating"],
            algorithm["k_factor"],
            algorithm["scale"],
            algorithm["rounding_policy"],
            document["processed_sequence"],
            document["processed_record_hash"],
            document["rated_results"],
            tuple(teams),
        )
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _decode(raw: bytes) -> BlindTeamRatingState:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("team_rating_encoding_invalid", "Team rating state has a BOM")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except _DuplicateJsonFieldError:
        _fail("team_rating_duplicate_field", "Team rating state has duplicate fields")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("team_rating_json_invalid", "Team rating state is malformed")
    return validate_team_rating_state_document(document)


def _encode(state: BlindTeamRatingState) -> bytes:
    validated = validate_team_rating_state_document(_state_document(state))
    return (
        json.dumps(
            _state_document(validated),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _ledger_results(ledger: BlindTeamResultLedgerState):
    if (
        not isinstance(ledger, BlindTeamResultLedgerState)
        or ledger.pending_result is not None
    ):
        _fail("team_rating_ledger_not_ready", "Team result ledger is not ready")
    return tuple(item.to_rating_result() for item in ledger.completed_results)


def _projection_status(
    state: BlindTeamRatingState,
    ledger: BlindTeamResultLedgerState,
) -> str:
    results = _ledger_results(ledger)
    validate_team_rating_state(state)
    if state.processed_sequence > len(results):
        _fail("team_rating_state_ahead", "Team rating state is ahead of history")
    expected_prefix = derive_team_rating_state(results[: state.processed_sequence])
    if state != expected_prefix:
        _fail("team_rating_state_diverged", "Team rating state diverges from history")
    return (
        TEAM_RATING_STATUS_SYNCED
        if state.processed_sequence == len(results)
        else TEAM_RATING_STATUS_BEHIND
    )


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamRatingSyncResult:
    state: BlindTeamRatingState = field(repr=False)
    status_before: str
    processed_before: int
    updates: tuple[BlindTeamRatingUpdate, ...]

    @property
    def applied_results(self) -> int:
        return self.state.processed_sequence - self.processed_before

    @property
    def last_update(self) -> BlindTeamRatingUpdate | None:
        return self.updates[-1] if self.updates else None

    def __repr__(self) -> str:
        return "BlindTeamRatingSyncResult(status_before={!r}, processed_results={!r})".format(
            self.status_before, self.state.processed_results
        )


class BlindTeamRatingStateStore:
    """Independent locked projection; the team result ledger is sole authority."""

    def __init__(
        self, config: BlindTeamRatingStateConfig, *, lock_timeout_seconds: float = 5.0
    ) -> None:
        if not isinstance(config, BlindTeamRatingStateConfig):
            _fail("team_rating_config_invalid", "Team rating configuration is invalid")
        self._config = config
        self._lock_timeout = lock_timeout_seconds

    def __repr__(self) -> str:
        return "BlindTeamRatingStateStore(configured=True)"

    def _load_unlocked(self, config: TeamStateFileConfig) -> BlindTeamRatingState:
        return _decode(stable_read_bytes(config))

    def load(self) -> BlindTeamRatingState:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            return self._load_unlocked(config)

    def initialize(self, ledger: BlindTeamResultLedgerState) -> BlindTeamRatingState:
        expected = derive_team_rating_state(_ledger_results(ledger))
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            if config.path.exists() or config.path.is_symlink():
                existing = self._load_unlocked(config)
                if existing != expected:
                    _fail(
                        "team_rating_initialization_conflict",
                        "Existing team ratings conflict",
                    )
                return existing
            write_atomic(config, _encode(expected), replace_existing=False)
        return expected

    def status(
        self, ledger: BlindTeamResultLedgerState
    ) -> tuple[str, BlindTeamRatingState]:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            return _projection_status(state, ledger), state

    def verify(self, ledger: BlindTeamResultLedgerState) -> BlindTeamRatingState:
        status, state = self.status(ledger)
        if status != TEAM_RATING_STATUS_SYNCED:
            _fail("team_rating_state_behind", "Team rating state is behind history")
        return state

    def sync(self, ledger: BlindTeamResultLedgerState) -> BlindTeamRatingSyncResult:
        results = _ledger_results(ledger)
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            status = _projection_status(state, ledger)
            processed_before = state.processed_sequence
            updates: list[BlindTeamRatingUpdate] = []
            if status == TEAM_RATING_STATUS_BEHIND:
                for result in results[state.processed_sequence :]:
                    state, update = apply_team_battle_result(state, result)
                    if update is not None:
                        updates.append(update)
                expected = derive_team_rating_state(results)
                if state != expected:
                    _fail("team_rating_state_diverged", "Team rating state diverges")
                write_atomic(config, _encode(state), replace_existing=True)
            return BlindTeamRatingSyncResult(
                state, status, processed_before, tuple(updates)
            )

    def rebuild(self, ledger: BlindTeamResultLedgerState) -> BlindTeamRatingState:
        expected = derive_team_rating_state(_ledger_results(ledger))
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            write_atomic(
                config,
                _encode(expected),
                replace_existing=config.path.exists() or config.path.is_symlink(),
            )
        return expected
