"""Explicit external configuration for private Blind Ladder artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .errors import BlindPoolValidationError
from .models import BlindPoolConfig, BlindPoolStateConfig


PRIVATE_ROOT_ENV = "TUGS_BLIND_POOL_ROOT"
REGISTRY_PATH_ENV = "TUGS_BLIND_POOL_REGISTRY"
STATE_PATH_ENV = "TUGS_BLIND_POOL_STATE"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _resolve_external_config_paths(
    config: BlindPoolConfig,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolConfig:
    try:
        private_root = config.private_root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise BlindPoolValidationError(
            "private_root_unavailable",
            "Blind Ladder private root is unavailable",
        ) from None
    if not private_root.is_dir():
        raise BlindPoolValidationError(
            "private_root_invalid",
            "Blind Ladder private root must be a directory",
        )

    try:
        registry_path = config.registry_path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise BlindPoolValidationError(
            "registry_file_missing",
            "Blind Ladder registry file is unavailable",
        ) from None
    if not registry_path.is_file():
        raise BlindPoolValidationError(
            "registry_file_invalid",
            "Blind Ladder registry must be a regular file",
        )

    repository = Path(repository_root or _repository_root()).resolve(strict=True)
    if _is_within(private_root, repository) or _is_within(repository, private_root):
        raise BlindPoolValidationError(
            "private_root_not_external",
            "Blind Ladder private root must be outside the repository",
        )
    if _is_within(registry_path, repository):
        raise BlindPoolValidationError(
            "registry_not_external",
            "Blind Ladder registry must be outside the repository",
        )

    return BlindPoolConfig(private_root, registry_path)


def load_blind_pool_config(
    environ: Mapping[str, str] | None = None,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolConfig | None:
    """Load optional external configuration without activating pool behavior."""

    values = os.environ if environ is None else environ
    root_value = values.get(PRIVATE_ROOT_ENV)
    registry_value = values.get(REGISTRY_PATH_ENV)
    if root_value is None and registry_value is None:
        return None
    if not root_value or not registry_value:
        raise BlindPoolValidationError(
            "config_incomplete",
            "Both Blind Ladder private configuration values are required",
        )
    if not isinstance(root_value, str) or not isinstance(registry_value, str):
        raise BlindPoolValidationError(
            "config_value_invalid",
            "Blind Ladder private configuration values must be path strings",
        )

    config = BlindPoolConfig(Path(root_value), Path(registry_value))
    return _resolve_external_config_paths(
        config,
        repository_root=repository_root,
    )


def validate_blind_pool_config(
    config: BlindPoolConfig,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolConfig:
    """Revalidate an explicitly constructed configuration before file access."""

    if not isinstance(config, BlindPoolConfig):
        raise BlindPoolValidationError(
            "config_type_invalid",
            "Blind Ladder configuration has an invalid type",
        )
    return _resolve_external_config_paths(
        config,
        repository_root=repository_root,
    )


def _resolve_state_path(
    config: BlindPoolStateConfig,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolStateConfig:
    pool_config = validate_blind_pool_config(
        config.pool_config,
        repository_root=repository_root,
    )
    state_path = config.state_path
    parent_path = state_path.parent
    try:
        if parent_path.exists() and not parent_path.is_dir():
            raise BlindPoolValidationError(
                "state_parent_invalid",
                "Blind Ladder state parent must be a directory",
            )
        parent = parent_path.resolve(strict=True)
    except BlindPoolValidationError:
        raise
    except (OSError, RuntimeError):
        raise BlindPoolValidationError(
            "state_parent_unavailable",
            "Blind Ladder state parent is unavailable",
        ) from None
    if not parent.is_dir():
        raise BlindPoolValidationError(
            "state_parent_invalid",
            "Blind Ladder state parent must be a directory",
        )

    try:
        if state_path.exists() or state_path.is_symlink():
            resolved_state = state_path.resolve(strict=True)
            if not resolved_state.is_file():
                raise BlindPoolValidationError(
                    "state_file_invalid",
                    "Blind Ladder state must be a regular file",
                )
        else:
            resolved_state = parent / state_path.name
    except BlindPoolValidationError:
        raise
    except (OSError, RuntimeError):
        raise BlindPoolValidationError(
            "state_file_invalid",
            "Blind Ladder state path is invalid",
        ) from None

    if not _is_within(resolved_state, pool_config.private_root):
        raise BlindPoolValidationError(
            "state_path_escape",
            "Blind Ladder state must remain beneath the private root",
        )
    repository = Path(repository_root or _repository_root()).resolve()
    if _is_within(resolved_state, repository):
        raise BlindPoolValidationError(
            "state_not_external",
            "Blind Ladder state must be outside the repository",
        )
    if resolved_state == pool_config.registry_path:
        raise BlindPoolValidationError(
            "state_registry_collision",
            "Blind Ladder state and registry must use different files",
        )

    lock_path = resolved_state.with_name(resolved_state.name + ".lock")
    try:
        resolved_lock = lock_path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise BlindPoolValidationError(
            "state_lock_path_invalid",
            "Blind Ladder state lock path is invalid",
        ) from None
    if not _is_within(resolved_lock, pool_config.private_root):
        raise BlindPoolValidationError(
            "state_lock_path_escape",
            "Blind Ladder state lock must remain beneath the private root",
        )
    if resolved_lock == pool_config.registry_path:
        raise BlindPoolValidationError(
            "state_lock_registry_collision",
            "Blind Ladder state lock and registry must use different files",
        )
    if lock_path.exists() or lock_path.is_symlink():
        try:
            if not resolved_lock.is_file():
                raise BlindPoolValidationError(
                    "state_lock_path_invalid",
                    "Blind Ladder state lock must be a regular file",
                )
        except OSError:
            raise BlindPoolValidationError(
                "state_lock_path_invalid",
                "Blind Ladder state lock path is invalid",
            ) from None

    return BlindPoolStateConfig(pool_config, resolved_state)


def load_blind_pool_state_config(
    environ: Mapping[str, str] | None = None,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolStateConfig | None:
    """Load optional bag-state configuration without changing legacy modes."""

    values = os.environ if environ is None else environ
    state_value = values.get(STATE_PATH_ENV)
    if state_value is None:
        return None
    if not isinstance(state_value, str) or not state_value:
        raise BlindPoolValidationError(
            "state_config_value_invalid",
            "Blind Ladder state configuration must be a path string",
        )
    if not values.get(PRIVATE_ROOT_ENV) or not values.get(REGISTRY_PATH_ENV):
        raise BlindPoolValidationError(
            "state_config_incomplete",
            "Blind Ladder state requires complete private registry configuration",
        )
    pool_config = load_blind_pool_config(
        values,
        repository_root=repository_root,
    )
    if pool_config is None:
        raise BlindPoolValidationError(
            "state_config_incomplete",
            "Blind Ladder state requires complete private registry configuration",
        )
    return _resolve_state_path(
        BlindPoolStateConfig(pool_config, Path(state_value)),
        repository_root=repository_root,
    )


def validate_blind_pool_state_config(
    config: BlindPoolStateConfig,
    *,
    repository_root: str | Path | None = None,
) -> BlindPoolStateConfig:
    """Revalidate an explicit state configuration before every transaction."""

    if not isinstance(config, BlindPoolStateConfig):
        raise BlindPoolValidationError(
            "state_config_invalid",
            "Blind Ladder state configuration has an invalid type",
        )
    return _resolve_state_path(config, repository_root=repository_root)
