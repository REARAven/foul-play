"""Network-free canonical deployment maintenance commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
import hmac
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
from .migration import migrate_blind_canonical_pool_expansion
from .ownership import (
    BlindPoolDeploymentOwnerGuard,
    acquire_blind_pool_deployment_owner,
)
from .reconciliation import (
    BlindReconciliationDisposition,
    derive_reconciliation_case,
    validate_reconciliation_case,
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
    reconciliation_case: str | None = field(default=None, repr=False)

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


@dataclass(frozen=True, slots=True, repr=False)
class BlindCanonicalReconciliationResult:
    """Content-neutral result of one explicit offline disposition."""

    disposition: BlindReconciliationDisposition

    def __repr__(self) -> str:
        return "BlindCanonicalReconciliationResult(disposition={!r})".format(
            self.disposition.value
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
    reconciliation_case: str | None = None
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
            case_invalid = False
            try:
                reconciliation_case = derive_reconciliation_case(state)
            except BlindPoolValidationError:
                case_invalid = True
            if case_invalid:
                _raise(
                    BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                    "blind_canonical_reconciliation_identity_invalid",
                    "Canonical accepted challenge identity is invalid",
                )
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
        reconciliation_case,
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


def _validated_state_config_for_owner(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None,
) -> BlindPoolStateConfig:
    """Validate only the paths needed to acquire deployment ownership."""

    if not isinstance(config, BlindCanonicalStartupConfig):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_maintenance_config_invalid",
            "Canonical maintenance configuration is invalid",
        )
    validation_failed = False
    try:
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
    except BlindPoolValidationError:
        validation_failed = True
        state_config = None
    if validation_failed:
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_deployment_invalid",
            "Canonical deployment validation failed",
        )
    assert state_config is not None
    return state_config


def _reconciliation_error(error_code: str) -> BlindCanonicalActivationError:
    if error_code in {
        "reconciliation_case_invalid",
        "reconciliation_case_mismatch",
    }:
        return _error(
            BlindCanonicalErrorCategory.RECONCILIATION_MISMATCH,
            "blind_canonical_reconciliation_case_mismatch",
            "Reconciliation case no longer matches current deployment state",
        )
    if error_code in {
        "state_not_initialized",
        "reservation_not_found",
        "reservation_phase_transition_invalid",
        "reconciliation_not_applicable",
    }:
        return _error(
            BlindCanonicalErrorCategory.RECONCILIATION_NOT_APPLICABLE,
            "blind_canonical_reconciliation_not_applicable",
            "Canonical deployment has no accepted challenge to reconcile",
        )
    if error_code == "registry_fingerprint_mismatch":
        return _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_canonical_state_registry_mismatch",
            "Canonical state requires explicit operator recovery",
        )
    return _error(
        BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
        "blind_canonical_reconciliation_failed",
        "Canonical reconciliation validation failed",
    )


def reconcile_blind_canonical_deployment(
    config: BlindCanonicalStartupConfig,
    expected_case: str,
    disposition: BlindReconciliationDisposition,
    *,
    repository_root: str | Path | None = None,
    owner_timeout_seconds: float = 0.25,
) -> BlindCanonicalReconciliationResult:
    """Apply one explicit offline disposition to the exact current incident."""

    case_error_code: str | None = None
    try:
        validated_case = validate_reconciliation_case(expected_case)
    except BlindPoolValidationError as error:
        case_error_code = error.code
        validated_case = None
    if case_error_code is not None:
        raise _reconciliation_error(case_error_code) from None
    if not isinstance(disposition, BlindReconciliationDisposition):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_reconciliation_disposition_invalid",
            "Canonical reconciliation disposition is invalid",
        )
    assert validated_case is not None

    owner_state_config = _validated_state_config_for_owner(
        config,
        repository_root=repository_root,
    )
    ownership_failed = False
    try:
        owner = acquire_blind_pool_deployment_owner(
            owner_state_config,
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

    operation_error: BlindCanonicalActivationError | None = None
    unexpected_error: BaseException | None = None
    result: BlindCanonicalReconciliationResult | None = None
    try:
        _registry, state_config, store = _load_immutable_deployment(
            config,
            repository_root=repository_root,
        )
        if state_config != owner_state_config:
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_deployment_changed",
                "Canonical deployment changed during reconciliation",
            )
        try:
            observed_state = store.snapshot()
            observed_case = derive_reconciliation_case(observed_state)
            if not hmac.compare_digest(validated_case, observed_case):
                raise BlindPoolValidationError(
                    "reconciliation_case_mismatch",
                    "Blind Ladder reconciliation case no longer matches current state",
                ) from None
            store.reconcile_accept_sent(validated_case, disposition)
        except BlindPoolValidationError as error:
            error_code = error.code
        else:
            error_code = None
        if error_code is not None:
            raise _reconciliation_error(error_code) from None
        result = BlindCanonicalReconciliationResult(disposition)
    except BlindCanonicalActivationError as error:
        operation_error = error
    except BaseException as error:
        unexpected_error = error
    finally:
        try:
            owner.close()
        except BlindPoolValidationError:
            operation_error = _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Canonical deployment ownership could not be released",
            )
    if unexpected_error is not None:
        raise unexpected_error
    if operation_error is not None:
        raise operation_error from None
    assert result is not None
    return result


def _add_deployment_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)


def _add_expansion_paths(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument("--{}-private-root".format(prefix), required=True)
    parser.add_argument("--{}-registry".format(prefix), required=True)
    parser.add_argument("--{}-state".format(prefix), required=True)


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
    _add_deployment_paths(commands.add_parser("status"))
    expansion = commands.add_parser("migrate-expansion")
    _add_expansion_paths(expansion, "source")
    _add_expansion_paths(expansion, "target")
    consumed = commands.add_parser(
        "resolve-consumed",
        help=(
            "Offline operator assertion that the quarantined selection should "
            "be counted as consumed"
        ),
    )
    _add_deployment_paths(consumed)
    consumed.add_argument("--case", required=True)
    consumed.add_argument("--confirm", required=True, choices=("consumed",))
    not_consumed = commands.add_parser(
        "resolve-not-consumed",
        help=(
            "Offline operator assertion that the quarantined selection should "
            "be released for retry"
        ),
    )
    _add_deployment_paths(not_consumed)
    not_consumed.add_argument("--case", required=True)
    not_consumed.add_argument(
        "--confirm",
        required=True,
        choices=("not-consumed",),
    )
    return parser


def _startup_config(args: argparse.Namespace) -> BlindCanonicalStartupConfig:
    return BlindCanonicalStartupConfig(
        Path(args.private_root),
        Path(args.registry),
        Path(args.state),
    )


def _expansion_config(
    args: argparse.Namespace,
    prefix: str,
) -> BlindCanonicalStartupConfig:
    return BlindCanonicalStartupConfig(
        Path(getattr(args, "{}_private_root".format(prefix))),
        Path(getattr(args, "{}_registry".format(prefix))),
        Path(getattr(args, "{}_state".format(prefix))),
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
        if args.command == "status":
            result = inspect_blind_canonical_deployment(_startup_config(args))
            print("deployment status: {}".format(result.status.value))
            if result.reconciliation_case is not None:
                print("reconciliation case: {}".format(result.reconciliation_case))
            return 0
        if args.command == "migrate-expansion":
            result = migrate_blind_canonical_pool_expansion(
                _expansion_config(args, "source"),
                _expansion_config(args, "target"),
            )
            print("expansion migration: complete")
            print("source active: {}".format(result.source_active_count))
            print("target active: {}".format(result.target_active_count))
            print(
                "consumed preserved: {}".format(
                    result.consumed_count_preserved,
                )
            )
            print("remaining: {}".format(result.remaining_count))
            print("target registry version: {}".format(result.target_registry_version))
            return 0
        if args.command == "resolve-consumed":
            result = reconcile_blind_canonical_deployment(
                _startup_config(args),
                args.case,
                BlindReconciliationDisposition.CONSUMED,
            )
            print("reconciliation resolved: {}".format(result.disposition.value))
            return 0
        if args.command == "resolve-not-consumed":
            result = reconcile_blind_canonical_deployment(
                _startup_config(args),
                args.case,
                BlindReconciliationDisposition.NOT_CONSUMED,
            )
            print("reconciliation resolved: {}".format(result.disposition.value))
            return 0
        raise AssertionError("unreachable maintenance command")
    except BlindCanonicalActivationError as error:
        print(
            "{}: {}".format(error.category.value, error.code),
            file=sys.stderr,
        )
        return 2
    except BlindPoolValidationError as error:
        print("migration: {}".format(error.code), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
