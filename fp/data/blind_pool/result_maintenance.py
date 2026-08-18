"""Privacy-safe offline maintenance for the Blind Ladder result ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .errors import BlindPoolValidationError
from .maintenance import _open_read_only_deployment
from .result_ledger import (
    OUTCOME_NO_RESULT,
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from .startup import BlindCanonicalActivationError, BlindCanonicalStartupConfig


_CLI_OUTCOMES = {
    "player-win": OUTCOME_PLAYER_WIN,
    "player-loss": OUTCOME_PLAYER_LOSS,
    "tie": OUTCOME_TIE,
    "no-result": OUTCOME_NO_RESULT,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Blind Ladder result-ledger maintenance",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "verify", "status", "recovery-case"):
        _add_paths(commands.add_parser(name))
    resolve = commands.add_parser("resolve")
    _add_paths(resolve)
    resolve.add_argument("--case", required=True)
    resolve.add_argument("--outcome", required=True, choices=tuple(_CLI_OUTCOMES))
    resolve.add_argument("--confirm", required=True, choices=tuple(_CLI_OUTCOMES))
    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--ledger", required=True)


def _startup_config(args: argparse.Namespace) -> BlindCanonicalStartupConfig:
    return BlindCanonicalStartupConfig(
        Path(args.private_root),
        Path(args.registry),
        Path(args.state),
        Path(args.ledger),
    )


def _run(args: argparse.Namespace) -> int:
    startup_config = _startup_config(args)
    opened = _open_read_only_deployment(
        startup_config,
        repository_root=None,
        owner_timeout_seconds=0.25,
    )
    operation_error: BlindPoolValidationError | None = None
    result = None
    try:
        result_config = validate_result_ledger_config(
            startup_config.result_ledger_path,
            private_root=startup_config.private_root,
            registry_path=startup_config.canonical_registry_path,
            selection_state_path=startup_config.state_path,
        )
        store = BlindResultLedgerStore(result_config)
        if args.command == "initialize":
            result = store.initialize_empty()
        elif args.command in {"verify", "status"}:
            result = store.load()
        elif args.command == "recovery-case":
            result = store.recovery_case()
        elif args.command == "resolve":
            if args.confirm != args.outcome:
                raise BlindPoolValidationError(
                    "result_recovery_confirmation_mismatch",
                    "Blind Ladder result recovery confirmation does not match",
                ) from None
            result = store.resolve_pending(args.case, _CLI_OUTCOMES[args.outcome])
    except BlindPoolValidationError as error:
        operation_error = error
    finally:
        try:
            opened.close()
        except BlindPoolValidationError as error:
            operation_error = error
    if operation_error is not None:
        raise operation_error from None

    if args.command == "initialize":
        print("result ledger: initialized")
        print("completed results: 0")
        print("pending results: 0")
    elif args.command == "verify":
        print("result ledger: verified")
        print("completed results: {}".format(result.completed_count))
        print("pending results: {}".format(result.pending_count))
    elif args.command == "status":
        status = "recovery_required" if result.pending_count else "ready"
        print("result ledger: {}".format(status))
        print("completed results: {}".format(result.completed_count))
        print("pending results: {}".format(result.pending_count))
    elif args.command == "recovery-case":
        print("result ledger: recovery_required")
        print("recovery case: {}".format(result))
    else:
        print("result recovery: complete")
        print("outcome: {}".format(args.outcome))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except BlindCanonicalActivationError as error:
        print(
            "{}: {}".format(error.category.value, error.code),
            file=sys.stderr,
        )
        return 2
    except BlindPoolValidationError as error:
        print("result_ledger: {}".format(error.code), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
