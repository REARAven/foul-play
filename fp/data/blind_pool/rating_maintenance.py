"""Privacy-safe offline maintenance for derived Blind Ladder Elo ratings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .errors import BlindPoolValidationError
from .lifecycle import normalize_showdown_identity
from .maintenance import _open_read_only_deployment
from .rating import baseline_player_rating
from .rating_state import (
    RATING_STATUS_BEHIND,
    BlindRatingStateStore,
    validate_rating_state_config,
)
from .result_ledger import BlindResultLedgerStore, validate_result_ledger_config
from .startup import BlindCanonicalActivationError, BlindCanonicalStartupConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Blind Ladder rating-state maintenance",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "verify", "status", "sync"):
        _add_paths(commands.add_parser(name))
    rebuild = commands.add_parser("rebuild")
    _add_paths(rebuild)
    rebuild.add_argument("--confirm", required=True, choices=("rebuild",))
    player = commands.add_parser("player-status")
    _add_paths(player)
    player.add_argument("--player", required=True)
    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--ratings", required=True)


def _startup_config(args: argparse.Namespace) -> BlindCanonicalStartupConfig:
    return BlindCanonicalStartupConfig(
        Path(args.private_root),
        Path(args.registry),
        Path(args.state),
        Path(args.ledger),
        Path(args.ratings),
    )


def _print_counts(state) -> None:
    print("processed results: {}".format(state.processed_results))
    print("rated results: {}".format(state.rated_results))


def _print_player(state, player_id: str) -> None:
    player = state.player(player_id) or baseline_player_rating(player_id)
    print("player rating: {}".format(player.rating))
    print("games: {}".format(player.games_played))
    print("wins: {}".format(player.wins))
    print("losses: {}".format(player.losses))
    print("ties: {}".format(player.ties))
    print("record: {}-{}-{}".format(player.wins, player.losses, player.ties))
    print("peak rating: {}".format(player.peak_rating))
    print("current streak: {}".format(player.streak_label))
    if player.games_played == 0:
        print("rating status: unrated")


def _run(args: argparse.Namespace) -> int:
    startup_config = _startup_config(args)
    opened = _open_read_only_deployment(
        startup_config,
        repository_root=None,
        owner_timeout_seconds=0.25,
    )
    operation_error: BlindPoolValidationError | None = None
    state = None
    status = None
    try:
        result_config = validate_result_ledger_config(
            startup_config.result_ledger_path,
            private_root=startup_config.private_root,
            registry_path=startup_config.canonical_registry_path,
            selection_state_path=startup_config.state_path,
        )
        result_state = BlindResultLedgerStore(result_config).require_ready()
        rating_config = validate_rating_state_config(
            startup_config.rating_state_path,
            private_root=startup_config.private_root,
            registry_path=startup_config.canonical_registry_path,
            selection_state_path=startup_config.state_path,
            result_ledger_path=startup_config.result_ledger_path,
        )
        store = BlindRatingStateStore(rating_config)
        if args.command == "initialize":
            state = store.initialize(result_state)
        elif args.command == "verify":
            state = store.verify(result_state)
        elif args.command == "status":
            status, state = store.status(result_state)
        elif args.command == "sync":
            synchronized = store.sync(result_state)
            state = synchronized.state
        elif args.command == "rebuild":
            state = store.rebuild(result_state)
        elif args.command == "player-status":
            player_id = normalize_showdown_identity(args.player)
            if not player_id:
                raise BlindPoolValidationError(
                    "rating_player_id_invalid",
                    "Player identity is invalid",
                ) from None
            state = store.verify(result_state)
            _print_player(state, player_id)
    except BlindPoolValidationError as error:
        operation_error = error
    finally:
        try:
            opened.close()
        except BlindPoolValidationError as error:
            operation_error = error
    if operation_error is not None:
        raise operation_error from None
    assert state is not None

    if args.command == "initialize":
        print("rating state: initialized")
        _print_counts(state)
    elif args.command == "verify":
        print("rating state: verified")
        _print_counts(state)
    elif args.command == "status":
        label = "behind" if status == RATING_STATUS_BEHIND else "ready"
        print("rating state: {}".format(label))
        _print_counts(state)
    elif args.command == "sync":
        print("rating state: synchronized")
        _print_counts(state)
    elif args.command == "rebuild":
        print("rating state: rebuilt")
        _print_counts(state)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except BlindCanonicalActivationError as error:
        print("{}: {}".format(error.category.value, error.code), file=sys.stderr)
        return 2
    except BlindPoolValidationError as error:
        print("rating_state: {}".format(error.code), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
