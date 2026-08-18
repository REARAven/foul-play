"""Explicit offline maintenance for dormant team-era Blind Ladder state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bag import BlindPoolBagStore
from .canonical_registry import load_canonical_runtime_registry
from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .leaderboard import build_leaderboard
from .models import BlindPoolConfig, BlindPoolStateConfig
from .ownership import acquire_blind_pool_deployment_owner
from .selection import create_canonical_selection_snapshot
from .state_migration import migrate_blind_pool_state_schema_2_to_3
from .team_public_registry import (
    PUBLIC_TEAM_KIND_BOT,
    PUBLIC_TEAM_KIND_PLAYER,
    BlindTeamPublicRegistryStore,
    validate_team_public_registry_config,
)
from .team_rating_state import (
    BlindTeamRatingStateStore,
    validate_team_rating_state_config,
)
from .team_result_ledger import (
    BlindTeamResultLedgerStore,
    validate_team_result_ledger_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "initialize",
            "migrate-selection-state",
            "preflight",
            "verify",
            "status",
            "sync",
            "leaderboard",
            "rebuild-ratings",
            "sync-bot-identities",
        ),
    )
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--canonical-registry", type=Path, required=True)
    parser.add_argument("--selection-state", type=Path, required=True)
    parser.add_argument("--team-result-ledger", type=Path, required=True)
    parser.add_argument("--team-rating-state", type=Path, required=True)
    parser.add_argument("--team-public-registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--confirm", choices=("rebuild",))
    return parser


def _stores(args: argparse.Namespace):
    public = BlindTeamPublicRegistryStore(
        validate_team_public_registry_config(
            args.team_public_registry,
            private_root=args.private_root,
            canonical_registry_path=args.canonical_registry,
            selection_state_path=args.selection_state,
            result_ledger_path=args.team_result_ledger,
            rating_state_path=args.team_rating_state,
            repository_root=args.repository_root,
        )
    )
    result = BlindTeamResultLedgerStore(
        validate_team_result_ledger_config(
            args.team_result_ledger,
            private_root=args.private_root,
            registry_path=args.canonical_registry,
            selection_state_path=args.selection_state,
            rating_state_path=args.team_rating_state,
            public_registry_path=args.team_public_registry,
            repository_root=args.repository_root,
        )
    )
    rating = BlindTeamRatingStateStore(
        validate_team_rating_state_config(
            args.team_rating_state,
            private_root=args.private_root,
            canonical_registry_path=args.canonical_registry,
            selection_state_path=args.selection_state,
            result_ledger_path=args.team_result_ledger,
            public_registry_path=args.team_public_registry,
            repository_root=args.repository_root,
        )
    )
    return public, result, rating


def _registry(args: argparse.Namespace):
    return load_canonical_runtime_registry(
        args.private_root,
        args.canonical_registry,
        repository_root=args.repository_root,
    )


def _verify_public(public_state, active_ids: Sequence[str]) -> None:
    for team_id in active_ids:
        identity = public_state.identity(team_id)
        if identity is None or identity.kind != "bot":
            raise BlindPoolValidationError(
                "team_public_registry_bot_missing",
                "Active bot public identity is missing",
            ) from None


def _selection_authority(args: argparse.Namespace):
    canonical = _registry(args)
    selection = create_canonical_selection_snapshot(canonical)
    state_config = validate_blind_pool_state_config(
        BlindPoolStateConfig(
            BlindPoolConfig(args.private_root, args.canonical_registry),
            args.selection_state,
        ),
        repository_root=args.repository_root,
    )
    store = BlindPoolBagStore.from_selection_snapshot(
        state_config,
        selection,
        team_mode=True,
    )
    return canonical, selection, state_config, store


def _verify_public_history(public_state, ledger) -> None:
    for record in ledger.completed_results:
        player = public_state.identity(record.player_team_id)
        bot = public_state.identity(record.bot_team_id)
        if (
            player is None
            or player.kind != PUBLIC_TEAM_KIND_PLAYER
            or bot is None
            or bot.kind != PUBLIC_TEAM_KIND_BOT
        ):
            raise BlindPoolValidationError(
                "team_public_registry_history_invalid",
                "Historical team identity is missing or invalid",
            ) from None


def _readiness(args: argparse.Namespace, public_store, result_store, rating_store):
    canonical, _selection, state_config, selection_store = _selection_authority(args)
    with acquire_blind_pool_deployment_owner(
        state_config,
        repository_root=args.repository_root,
    ):
        selection_state = selection_store.snapshot()
        if selection_state.reservation is not None:
            raise BlindPoolValidationError(
                "team_selection_recovery_required",
                "Team selection recovery is required",
            ) from None
        public_state = public_store.load()
        _verify_public(public_state, canonical.active_ids)
        ledger = result_store.require_ready()
        _verify_public_history(public_state, ledger)
        rating_state = rating_store.verify(ledger)
    return selection_state, public_state, ledger, rating_state


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    public_store, result_store, rating_store = _stores(args)
    command = args.command

    if command == "migrate-selection-state":
        _canonical, selection, state_config, team_store = _selection_authority(args)
        with acquire_blind_pool_deployment_owner(
            state_config,
            repository_root=args.repository_root,
        ):
            try:
                state = team_store.snapshot()
            except BlindPoolValidationError as error:
                if error.code != "state_schema_unsupported":
                    raise
                state = migrate_blind_pool_state_schema_2_to_3(
                    state_config,
                    selection,
                )
                migration_status = "complete"
            else:
                migration_status = "already complete"
        print("selection state migration: {}".format(migration_status))
        print("selection schema: {}".format(state.schema_version))
        print("reservation pending: {}".format(int(state.reservation is not None)))
        return 0

    if command == "initialize":
        initialization_paths = (
            args.team_public_registry,
            args.team_result_ledger,
            args.team_rating_state,
        )
        if any(path.exists() or path.is_symlink() for path in initialization_paths):
            raise BlindPoolValidationError(
                "team_ladder_initialization_conflict",
                "Team ladder state already exists",
            ) from None
        canonical = _registry(args)
        public = public_store.initialize(canonical.active_ids)
        ledger = result_store.initialize_empty()
        rating = rating_store.initialize(ledger)
        print("team public registry: initialized")
        print("bot identities registered: {}".format(public.bot_count))
        print("player identities registered: {}".format(public.player_count))
        print("team result ledger: initialized")
        print("completed results: {}".format(ledger.completed_count))
        print("pending results: {}".format(ledger.pending_count))
        print("team rating state: initialized")
        print("processed results: {}".format(rating.processed_sequence))
        print("rated results: {}".format(rating.rated_results))
        return 0

    if command == "sync-bot-identities":
        canonical = _registry(args)
        state = public_store.sync_bot_identities(canonical.active_ids)
        print("bot identities registered: {}".format(state.bot_count))
        return 0

    if command in {"preflight", "verify", "status"}:
        selection_state, public, ledger, state = _readiness(
            args,
            public_store,
            result_store,
            rating_store,
        )
        if command == "preflight":
            print("team ladder preflight: ready")
            print("selection schema: {}".format(selection_state.schema_version))
        elif command == "verify":
            print("team ladder state: verified")
            return 0
        print("completed results: {}".format(ledger.completed_count))
        print("pending results: {}".format(ledger.pending_count))
        print("processed results: {}".format(state.processed_sequence))
        print("rated results: {}".format(state.rated_results))
        print("bot identities registered: {}".format(public.bot_count))
        print("player identities registered: {}".format(public.player_count))
        return 0

    public = public_store.load()
    ledger = result_store.require_ready()
    if command == "rebuild-ratings":
        if args.confirm != "rebuild":
            raise RuntimeError("rebuild confirmation is required")
        state = rating_store.rebuild(ledger)
        print("team rating state: rebuilt")
        print("processed results: {}".format(state.processed_sequence))
        return 0
    if command == "sync":
        result = rating_store.sync(ledger)
        print("team rating state: synced")
        print("processed results: {}".format(result.state.processed_sequence))
        return 0

    state = rating_store.verify(ledger)
    rows = [
        dict(row.to_public_dict())
        for row in build_leaderboard(state, public.public_registry)
    ]
    print(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
