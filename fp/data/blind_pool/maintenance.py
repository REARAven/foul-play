"""Network-free canonical deployment maintenance commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import NoReturn, Sequence

from .bag import BlindPoolBagStore
from .canonical_artifacts import load_canonical_team_artifact
from .canonical_models import (
    CanonicalArtifactError,
    CanonicalRuntimeRegistry,
)
from .canonical_registry import load_canonical_runtime_registry
from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .models import BlindPoolConfig, BlindPoolStateConfig
from .ownership import (
    BlindPoolDeploymentOwnerGuard,
    acquire_blind_pool_deployment_owner,
)
from .registry_generation import (
    CanonicalRegistryBuildConfig,
    build_canonical_runtime_registry,
)
from .selection import create_canonical_selection_snapshot
from .startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    BlindCanonicalStartupConfig,
)
from .state import (
    ACCEPT_SENT_PHASE,
    RESERVATION_PHASE,
    load_blind_pool_bag_state,
)


class BlindCanonicalInspectionStatus(str, Enum):
    READY_FOR_FIRST_START = "ready_for_first_start"
    READY = "ready"
    STARTUP_RECOVERY_AVAILABLE = "startup_recovery_available"
    RECOVERY_REQUIRED = "recovery_required"


@dataclass(frozen=True, slots=True, repr=False)
class BlindCanonicalInspectionResult:
    """Privacy-safe aggregate state of one offline deployment inspection."""

    status: BlindCanonicalInspectionStatus
    registry_version: str
    entry_count: int
    active_count: int

    @property
    def ready(self) -> bool:
        return self.status is not BlindCanonicalInspectionStatus.RECOVERY_REQUIRED

    def __repr__(self) -> str:
        return (
            "BlindCanonicalInspectionResult(status={!r}, "
            "registry_version={!r}, entry_count={!r}, active_count={!r})"
        ).format(
            self.status.value,
            self.registry_version,
            self.entry_count,
            self.active_count,
        )


@dataclass(frozen=True, slots=True, repr=False)
class BlindCanonicalArtifactVerificationResult:
    """Safe aggregate result of active-only full artifact verification."""

    active_count: int

    def __repr__(self) -> str:
        return "BlindCanonicalArtifactVerificationResult(active_count={!r})".format(
            self.active_count
        )


@dataclass(slots=True, repr=False)
class _OpenReadOnlyDeployment:
    registry: CanonicalRuntimeRegistry = field(repr=False)
    state_config: BlindPoolStateConfig = field(repr=False)
    owner: BlindPoolDeploymentOwnerGuard = field(repr=False)
    result: BlindCanonicalInspectionResult

    def close(self) -> None:
        self.owner.close()


def _error(
    category: BlindCanonicalErrorCategory,
    code: str,
    message: str,
) -> BlindCanonicalActivationError:
    return BlindCanonicalActivationError(category, code, message)


def _raise(
    category: BlindCanonicalErrorCategory,
    code: str,
    message: str,
) -> NoReturn:
    raise _error(category, code, message) from None


def _load_immutable_deployment(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None,
) -> tuple[CanonicalRuntimeRegistry, BlindPoolStateConfig, BlindPoolBagStore]:
    if not isinstance(config, BlindCanonicalStartupConfig):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_maintenance_config_invalid",
            "Canonical maintenance configuration is invalid",
        )
    failure_category: BlindCanonicalErrorCategory | None = None
    try:
        registry = load_canonical_runtime_registry(
            config.private_root,
            config.canonical_registry_path,
            repository_root=repository_root,
        )
        selection = create_canonical_selection_snapshot(registry)
        state_config = validate_blind_pool_state_config(
            BlindPoolStateConfig(
                BlindPoolConfig(
                    config.private_root,
                    config.canonical_registry_path,
                ),
                config.state_path,
            ),
            repository_root=repository_root,
        )
        store = BlindPoolBagStore.from_selection_snapshot(
            state_config,
            selection,
        )
    except CanonicalArtifactError:
        failure_category = BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY
        registry = state_config = store = None
    except BlindPoolValidationError:
        failure_category = BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY
        registry = state_config = store = None
    if failure_category is not None:
        _raise(
            failure_category,
            "blind_canonical_deployment_invalid",
            "Canonical deployment validation failed",
        )
    assert registry is not None and state_config is not None and store is not None
    return registry, state_config, store


def _inspect_state_read_only(
    registry: CanonicalRuntimeRegistry,
    store: BlindPoolBagStore,
    state_config: BlindPoolStateConfig,
) -> BlindCanonicalInspectionResult:
    if not state_config.state_path.exists():
        status = BlindCanonicalInspectionStatus.READY_FOR_FIRST_START
    else:
        state_error_code: str | None = None
        try:
            state = load_blind_pool_bag_state(
                state_config,
                store._selection,
            )
        except BlindPoolValidationError as error:
            state_error_code = error.code
            state = None
        if state_error_code == "registry_fingerprint_mismatch":
            _raise(
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                "blind_canonical_state_registry_mismatch",
                "Canonical state requires explicit operator recovery",
            )
        if state_error_code is not None:
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_state_invalid",
                "Canonical state is invalid",
            )
        assert state is not None
        if state.reservation is None:
            status = BlindCanonicalInspectionStatus.READY
        elif state.reservation.phase == RESERVATION_PHASE:
            status = BlindCanonicalInspectionStatus.STARTUP_RECOVERY_AVAILABLE
        elif state.reservation.phase == ACCEPT_SENT_PHASE:
            status = BlindCanonicalInspectionStatus.RECOVERY_REQUIRED
        else:
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_state_invalid",
                "Canonical state is invalid",
            )
    return BlindCanonicalInspectionResult(
        status,
        registry.registry_version,
        len(registry.entries),
        len(registry.active_ids),
    )


def _open_read_only_deployment(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None,
    owner_timeout_seconds: float,
) -> _OpenReadOnlyDeployment:
    registry, state_config, store = _load_immutable_deployment(
        config,
        repository_root=repository_root,
    )
    ownership_failed = False
    try:
        owner = acquire_blind_pool_deployment_owner(
            state_config,
            timeout_seconds=owner_timeout_seconds,
            repository_root=repository_root,
        )
    except BlindPoolValidationError:
        ownership_failed = True
        owner = None
    if ownership_failed:
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            "blind_canonical_deployment_in_use",
            "Canonical deployment is already active or unavailable",
        )
    assert owner is not None

    inspection_error: BlindCanonicalActivationError | None = None
    result: BlindCanonicalInspectionResult | None = None
    try:
        result = _inspect_state_read_only(registry, store, state_config)
    except BlindCanonicalActivationError as error:
        inspection_error = error
    except BaseException:
        try:
            owner.close()
        finally:
            raise
    if inspection_error is not None:
        try:
            owner.close()
        except BlindPoolValidationError:
            inspection_error = _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Canonical deployment ownership could not be released",
            )
        raise inspection_error from None
    assert result is not None
    return _OpenReadOnlyDeployment(registry, state_config, owner, result)


def inspect_blind_canonical_deployment(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None = None,
    owner_timeout_seconds: float = 0.25,
) -> BlindCanonicalInspectionResult:
    """Inspect immutable deployment and state without initialization or recovery."""

    opened = _open_read_only_deployment(
        config,
        repository_root=repository_root,
        owner_timeout_seconds=owner_timeout_seconds,
    )
    close_error = None
    try:
        result = opened.result
    finally:
        try:
            opened.close()
        except BlindPoolValidationError:
            close_error = _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Canonical deployment ownership could not be released",
            )
    if close_error is not None:
        raise close_error from None
    return result


def verify_active_blind_canonical_artifacts(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None = None,
    owner_timeout_seconds: float = 0.25,
) -> BlindCanonicalArtifactVerificationResult:
    """Fully verify active artifacts one at a time under deployment ownership."""

    opened = _open_read_only_deployment(
        config,
        repository_root=repository_root,
        owner_timeout_seconds=owner_timeout_seconds,
    )
    operation_error: BlindCanonicalActivationError | None = None
    verified = 0
    try:
        for team_id in sorted(opened.registry.active_ids):
            artifact_failed = False
            try:
                artifact = load_canonical_team_artifact(opened.registry, team_id)
            except CanonicalArtifactError:
                artifact_failed = True
                artifact = None
            if artifact_failed:
                _raise(
                    BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                    "blind_canonical_artifact_verification_failed",
                    "Active canonical artifact verification failed",
                )
            assert artifact is not None
            verified += 1
            del artifact
    except BlindCanonicalActivationError as error:
        operation_error = error
    finally:
        try:
            opened.close()
        except BlindPoolValidationError:
            operation_error = _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Canonical deployment ownership could not be released",
            )
    if operation_error is not None:
        raise operation_error from None
    return BlindCanonicalArtifactVerificationResult(verified)


def _add_deployment_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline canonical Blind Ladder deployment maintenance",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-registry")
    _add_deployment_paths(build)
    build.add_argument("--plan", required=True)
    _add_deployment_paths(commands.add_parser("preflight"))
    _add_deployment_paths(commands.add_parser("verify-artifacts"))
    return parser


def _startup_config(args: argparse.Namespace) -> BlindCanonicalStartupConfig:
    return BlindCanonicalStartupConfig(
        Path(args.private_root),
        Path(args.registry),
        Path(args.state),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit offline command and return a stable process result."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "build-registry":
            result = build_canonical_runtime_registry(
                CanonicalRegistryBuildConfig(
                    Path(args.private_root),
                    Path(args.plan),
                    Path(args.registry),
                    Path(args.state),
                )
            )
            print("registry generated")
            print("registry version: {}".format(result.registry_version))
            print("entries: {}".format(result.entry_count))
            print("active: {}".format(result.active_count))
            return 0
        if args.command == "preflight":
            result = inspect_blind_canonical_deployment(_startup_config(args))
            print("deployment preflight: {}".format(result.status.value))
            return 0 if result.ready else 2
        if args.command == "verify-artifacts":
            result = verify_active_blind_canonical_artifacts(_startup_config(args))
            print("active artifacts verified: {}".format(result.active_count))
            return 0
        raise AssertionError("unreachable maintenance command")
    except BlindCanonicalActivationError as error:
        print(
            "{}: {}".format(error.category.value, error.code),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
