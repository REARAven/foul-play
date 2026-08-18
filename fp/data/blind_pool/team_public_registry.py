"""Private durable mapping from stable team identities to public ladder names."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, NoReturn, Protocol, Sequence

from .errors import BlindPoolValidationError
from .leaderboard import (
    BlindTeamPublicIdentity,
    BlindTeamPublicRegistry,
    PUBLIC_TEAM_KIND_BOT,
    PUBLIC_TEAM_KIND_PLAYER,
)
from .locking import BlindPoolStateLock
from .models import is_valid_opaque_team_id, is_valid_player_team_id
from .team_persistence import (
    TeamStateFileConfig,
    stable_read_bytes,
    validate_team_state_file_config,
    write_atomic,
)


TEAM_PUBLIC_REGISTRY_SCHEMA_VERSION = 1
TEAM_PUBLIC_REGISTRY_PATH_ENV = "TUGS_BLIND_TEAM_PUBLIC_REGISTRY"
_TOP_FIELDS = frozenset({"schema_version", "next_bot_alias_number", "identities"})
_IDENTITY_FIELDS = frozenset({"private_team_id", "display_name", "kind"})


class ShuffleSource(Protocol):
    def shuffle(self, values: list[str]) -> None: ...


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamPublicRegistryConfig:
    registry_path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    canonical_registry_path: Path = field(repr=False)
    selection_state_path: Path = field(repr=False)
    result_ledger_path: Path | None = field(default=None, repr=False)
    rating_state_path: Path | None = field(default=None, repr=False)
    repository_root: Path | None = field(default=None, repr=False)

    @property
    def lock_path(self) -> Path:
        return self.registry_path.with_name(self.registry_path.name + ".lock")

    def __repr__(self) -> str:
        return "BlindTeamPublicRegistryConfig(configured=True)"


def validate_team_public_registry_config(
    registry_path: str | Path,
    *,
    private_root: str | Path,
    canonical_registry_path: str | Path,
    selection_state_path: str | Path,
    result_ledger_path: str | Path | None = None,
    rating_state_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> BlindTeamPublicRegistryConfig:
    collisions = [Path(canonical_registry_path), Path(selection_state_path)]
    if result_ledger_path is not None:
        collisions.append(Path(result_ledger_path))
    if rating_state_path is not None:
        collisions.append(Path(rating_state_path))
    validated = validate_team_state_file_config(
        registry_path,
        private_root=private_root,
        collision_paths=tuple(collisions),
        code_prefix="team_public_registry",
        repository_root=repository_root,
    )
    return BlindTeamPublicRegistryConfig(
        validated.path,
        validated.private_root,
        Path(canonical_registry_path).resolve(strict=False),
        Path(selection_state_path).resolve(strict=False),
        None
        if result_ledger_path is None
        else Path(result_ledger_path).resolve(strict=False),
        None
        if rating_state_path is None
        else Path(rating_state_path).resolve(strict=False),
        validated.repository_root,
    )


def _file_config(config: BlindTeamPublicRegistryConfig) -> TeamStateFileConfig:
    collisions = [config.canonical_registry_path, config.selection_state_path]
    if config.result_ledger_path is not None:
        collisions.append(config.result_ledger_path)
    if config.rating_state_path is not None:
        collisions.append(config.rating_state_path)
    return validate_team_state_file_config(
        config.registry_path,
        private_root=config.private_root,
        collision_paths=tuple(collisions),
        code_prefix="team_public_registry",
        repository_root=config.repository_root,
    )


@dataclass(frozen=True, slots=True, repr=False)
class BlindTeamPublicRegistryState:
    schema_version: int
    next_bot_alias_number: int
    identities: tuple[BlindTeamPublicIdentity, ...] = field(repr=False)

    @property
    def public_registry(self) -> BlindTeamPublicRegistry:
        return BlindTeamPublicRegistry(self.identities)

    @property
    def bot_count(self) -> int:
        return sum(item.kind == PUBLIC_TEAM_KIND_BOT for item in self.identities)

    @property
    def player_count(self) -> int:
        return sum(item.kind == PUBLIC_TEAM_KIND_PLAYER for item in self.identities)

    def identity(self, team_id: str) -> BlindTeamPublicIdentity | None:
        return self.public_registry.identity(team_id)

    def __repr__(self) -> str:
        return "BlindTeamPublicRegistryState(bot_count={!r}, player_count={!r})".format(
            self.bot_count, self.player_count
        )


def _state_document(state: BlindTeamPublicRegistryState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "next_bot_alias_number": state.next_bot_alias_number,
        "identities": [
            {
                "private_team_id": item.team_id,
                "display_name": item.display_name,
                "kind": item.kind,
            }
            for item in state.identities
        ],
    }


def validate_team_public_registry_document(
    document: Any,
) -> BlindTeamPublicRegistryState:
    if not isinstance(document, dict) or set(document) != _TOP_FIELDS:
        _fail(
            "team_public_registry_document_invalid", "Team public registry is invalid"
        )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != TEAM_PUBLIC_REGISTRY_SCHEMA_VERSION
    ):
        _fail(
            "team_public_registry_schema_unsupported",
            "Team public registry schema is unsupported",
        )
    next_number = document["next_bot_alias_number"]
    if type(next_number) is not int or next_number < 1:
        _fail(
            "team_public_registry_alias_counter_invalid", "Bot alias counter is invalid"
        )
    raw_identities = document["identities"]
    if not isinstance(raw_identities, list):
        _fail(
            "team_public_registry_document_invalid",
            "Team public identities are invalid",
        )
    identities: list[BlindTeamPublicIdentity] = []
    issued_numbers: set[int] = set()
    for raw in raw_identities:
        if not isinstance(raw, dict) or set(raw) != _IDENTITY_FIELDS:
            _fail(
                "team_public_registry_fields_invalid",
                "Team public identity fields are invalid",
            )
        identity = BlindTeamPublicIdentity(
            raw["private_team_id"], raw["display_name"], raw["kind"]
        )
        if identity.kind == PUBLIC_TEAM_KIND_BOT:
            if not is_valid_opaque_team_id(identity.team_id):
                _fail("team_public_registry_bot_id_invalid", "Bot identity is invalid")
            prefix = "Bot team "
            if not identity.display_name.startswith(prefix):
                _fail("team_public_registry_bot_alias_invalid", "Bot alias is invalid")
            suffix = identity.display_name[len(prefix) :]
            if not suffix.isascii() or not suffix.isdigit() or int(suffix) < 1:
                _fail("team_public_registry_bot_alias_invalid", "Bot alias is invalid")
            number = int(suffix)
            if identity.display_name != "Bot team {:02d}".format(number):
                _fail("team_public_registry_bot_alias_invalid", "Bot alias is invalid")
            issued_numbers.add(number)
        elif not is_valid_player_team_id(identity.team_id):
            _fail(
                "team_public_registry_player_id_invalid", "Player identity is invalid"
            )
        identities.append(identity)
    registry = BlindTeamPublicRegistry(tuple(identities))
    if issued_numbers and max(issued_numbers) >= next_number:
        _fail(
            "team_public_registry_alias_counter_invalid",
            "Bot alias counter is inconsistent",
        )
    return BlindTeamPublicRegistryState(
        TEAM_PUBLIC_REGISTRY_SCHEMA_VERSION,
        next_number,
        registry.identities,
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _decode(raw: bytes) -> BlindTeamPublicRegistryState:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("team_public_registry_encoding_invalid", "Team public registry has a BOM")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except _DuplicateJsonFieldError:
        _fail(
            "team_public_registry_duplicate_field",
            "Team public registry has duplicate fields",
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("team_public_registry_json_invalid", "Team public registry is malformed")
    return validate_team_public_registry_document(document)


def _encode(state: BlindTeamPublicRegistryState) -> bytes:
    validated = validate_team_public_registry_document(_state_document(state))
    return (
        json.dumps(
            _state_document(validated),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class BlindTeamPublicRegistryStore:
    """Locked persistence for stable public aliases; repr never exposes mapping."""

    def __init__(
        self,
        config: BlindTeamPublicRegistryConfig,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(config, BlindTeamPublicRegistryConfig):
            _fail(
                "team_public_registry_config_invalid",
                "Team public registry configuration is invalid",
            )
        self._config = config
        self._lock_timeout = lock_timeout_seconds

    def __repr__(self) -> str:
        return "BlindTeamPublicRegistryStore(configured=True)"

    def _load_unlocked(
        self, config: TeamStateFileConfig
    ) -> BlindTeamPublicRegistryState:
        return _decode(stable_read_bytes(config))

    def load(self) -> BlindTeamPublicRegistryState:
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            return self._load_unlocked(config)

    def initialize(
        self,
        active_bot_team_ids: Sequence[str],
        *,
        random_source: ShuffleSource | None = None,
    ) -> BlindTeamPublicRegistryState:
        if not isinstance(active_bot_team_ids, Sequence) or isinstance(
            active_bot_team_ids, (str, bytes, bytearray)
        ):
            _fail("team_public_registry_bot_ids_invalid", "Bot identities are invalid")
        bot_ids = list(active_bot_team_ids)
        if len(set(bot_ids)) != len(bot_ids) or any(
            not is_valid_opaque_team_id(item) for item in bot_ids
        ):
            _fail("team_public_registry_bot_ids_invalid", "Bot identities are invalid")
        source = random.SystemRandom() if random_source is None else random_source
        if not callable(getattr(source, "shuffle", None)):
            _fail(
                "team_public_registry_random_invalid", "Bot alias randomness is invalid"
            )
        source.shuffle(bot_ids)
        identities = tuple(
            BlindTeamPublicIdentity(
                team_id,
                "Bot team {:02d}".format(number),
                PUBLIC_TEAM_KIND_BOT,
            )
            for number, team_id in enumerate(bot_ids, start=1)
        )
        state = validate_team_public_registry_document(
            _state_document(
                BlindTeamPublicRegistryState(
                    TEAM_PUBLIC_REGISTRY_SCHEMA_VERSION,
                    len(bot_ids) + 1,
                    identities,
                )
            )
        )
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            if config.path.exists() or config.path.is_symlink():
                _fail(
                    "team_public_registry_target_exists",
                    "Team public registry already exists",
                )
            write_atomic(config, _encode(state), replace_existing=False)
        return state

    def register_player(
        self,
        identity: BlindTeamPublicIdentity,
    ) -> BlindTeamPublicRegistryState:
        if (
            not isinstance(identity, BlindTeamPublicIdentity)
            or identity.kind != PUBLIC_TEAM_KIND_PLAYER
        ):
            _fail("team_public_registry_player_invalid", "Player identity is invalid")
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            existing = state.identity(identity.team_id)
            if existing is not None and existing.kind != PUBLIC_TEAM_KIND_PLAYER:
                _fail(
                    "team_public_registry_kind_conflict", "Team identity kind conflicts"
                )
            identities = tuple(
                item for item in state.identities if item.team_id != identity.team_id
            ) + (identity,)
            updated = validate_team_public_registry_document(
                _state_document(
                    BlindTeamPublicRegistryState(
                        state.schema_version, state.next_bot_alias_number, identities
                    )
                )
            )
            if updated != state:
                write_atomic(config, _encode(updated), replace_existing=True)
            return updated

    def sync_bot_identities(
        self,
        active_bot_team_ids: Sequence[str],
        *,
        random_source: ShuffleSource | None = None,
    ) -> BlindTeamPublicRegistryState:
        if not isinstance(active_bot_team_ids, Sequence) or isinstance(
            active_bot_team_ids, (str, bytes, bytearray)
        ):
            _fail("team_public_registry_bot_ids_invalid", "Bot identities are invalid")
        active = list(active_bot_team_ids)
        if len(set(active)) != len(active) or any(
            not is_valid_opaque_team_id(item) for item in active
        ):
            _fail("team_public_registry_bot_ids_invalid", "Bot identities are invalid")
        source = random.SystemRandom() if random_source is None else random_source
        if not callable(getattr(source, "shuffle", None)):
            _fail(
                "team_public_registry_random_invalid", "Bot alias randomness is invalid"
            )
        config = _file_config(self._config)
        with BlindPoolStateLock(config.lock_path, timeout_seconds=self._lock_timeout):
            state = self._load_unlocked(config)
            existing_ids = {item.team_id for item in state.identities}
            additions = [item for item in active if item not in existing_ids]
            source.shuffle(additions)
            identities = list(state.identities)
            next_number = state.next_bot_alias_number
            for team_id in additions:
                identities.append(
                    BlindTeamPublicIdentity(
                        team_id,
                        "Bot team {:02d}".format(next_number),
                        PUBLIC_TEAM_KIND_BOT,
                    )
                )
                next_number += 1
            updated = validate_team_public_registry_document(
                _state_document(
                    BlindTeamPublicRegistryState(
                        state.schema_version, next_number, tuple(identities)
                    )
                )
            )
            if updated != state:
                write_atomic(config, _encode(updated), replace_existing=True)
            return updated
