"""Connection-scoped orchestration for explicitly prepared canonical mode."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Mapping

from requests.exceptions import RequestException
from websockets.exceptions import WebSocketException

from fp.websocket_client import LocalLoginConfigurationError, LoginError

from .errors import BlindPoolLifecycleError, BlindPoolValidationError
from .startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    classify_blind_canonical_runtime_error,
    load_blind_canonical_startup_config,
    prepare_blind_canonical_deployment,
)


logger = logging.getLogger(__name__)

_EXPECTED_NETWORK_ERRORS = (
    IndexError,
    KeyError,
    LocalLoginConfigurationError,
    LoginError,
    OSError,
    RequestException,
    TimeoutError,
    WebSocketException,
    json.JSONDecodeError,
)


async def _run_sequential_battles(
    client: object,
    prepared_deployment: object,
    process_config: object,
    public_prior_configuration: object,
    original_pokedex: object,
    original_move_json: object,
    battle_runner: Callable[..., Awaitable[Any]],
    integrity_checker: Callable[[object, object], None],
) -> None:
    async def initialize_battle(_room, team_projection):
        runtime = runtime_holder[0]
        if runtime is None:
            raise BlindPoolLifecycleError(
                "result_runtime_unavailable",
                "Blind canonical result runtime is unavailable",
            ) from None
        return await battle_runner(
            client,
            process_config.pokemon_format,
            team_projection,
            public_prior_configuration=public_prior_configuration,
            terminal_result_handler=runtime.record_terminal_result,
        )

    runtime_holder: list[object | None] = [None]
    runtime_error = None
    try:
        runtime = prepared_deployment.create_runtime(
            client,
            client.update_team,
            initialize_battle,
        )
    except (BlindPoolLifecycleError, BlindPoolValidationError) as error:
        runtime_error = error
        runtime = None
    if runtime_error is not None:
        raise classify_blind_canonical_runtime_error(runtime_error) from None
    assert runtime is not None
    runtime_holder[0] = runtime

    battles_run = 0
    while True:
        runtime_error = None
        try:
            await runtime.run_once()
        except (BlindPoolLifecycleError, BlindPoolValidationError) as error:
            runtime_error = error
        if runtime_error is not None:
            raise classify_blind_canonical_runtime_error(runtime_error) from None

        team_update = getattr(runtime, "last_team_rating_update", None)
        update = runtime.last_rating_update
        if team_update is not None:
            logger.info(
                "Team rating: {} {} ({:+d})".format(
                    team_update.player_name,
                    team_update.player_rating_after,
                    team_update.player_delta,
                )
            )
            logger.info(
                "Record: {}-{}-{} | Win rate: {}".format(
                    team_update.player_wins,
                    team_update.player_losses,
                    team_update.player_ties,
                    team_update.player_win_rate,
                )
            )
            logger.info("Current streak: {}".format(team_update.player_streak))
            logger.info("Peak rating: {}".format(team_update.player_peak))
            logger.info(
                "Opponent: {} | Rating: {} | Record: {}-{}-{} | Win rate: {}".format(
                    team_update.bot_name,
                    team_update.bot_rating_after,
                    team_update.bot_wins,
                    team_update.bot_losses,
                    team_update.bot_ties,
                    team_update.bot_win_rate,
                )
            )
        elif update is None:
            raise classify_blind_canonical_runtime_error(
                BlindPoolLifecycleError(
                    "canonical_rating_update_missing",
                    "Blind Ladder persisted rating update is unavailable",
                )
            ) from None
        else:
            logger.info(
                "Ladder rating: {} ({:+d})".format(
                    update.rating_after,
                    update.rating_delta,
                )
            )
            logger.info(
                "Record: {}-{}-{}".format(update.wins, update.losses, update.ties)
            )
            logger.info("Current streak: {}".format(update.streak_label))
            logger.info("Peak rating: {}".format(update.peak_rating))
        integrity_checker(original_pokedex, original_move_json)

        battles_run += 1
        if battles_run >= process_config.run_count:
            return


async def run_blind_canonical_activation(
    process_config: object,
    public_prior_configuration: object,
    original_pokedex: object,
    original_move_json: object,
    *,
    websocket_factory: Callable[..., Awaitable[object]],
    environ: Mapping[str, str] | None,
    battle_runner: Callable[..., Awaitable[Any]],
    integrity_checker: Callable[[object, object], None],
) -> None:
    """Prepare locally, then own one exact sequential connection/battle loop."""

    logger.info("Team source: blind-canonical")
    prepared_deployment = None
    client = None
    activation_error = None
    try:
        startup_config = load_blind_canonical_startup_config(environ)
        prepared_deployment = prepare_blind_canonical_deployment(startup_config)

        network_failed = False
        try:
            client = await websocket_factory(
                process_config.username,
                process_config.password,
                process_config.websocket_uri,
                process_config.local_no_security_login,
            )
            process_config.user_id = await client.login()
            if process_config.avatar is not None:
                await client.avatar(process_config.avatar)
        except _EXPECTED_NETWORK_ERRORS:
            network_failed = True
        if network_failed:
            raise BlindCanonicalActivationError(
                BlindCanonicalErrorCategory.TRANSIENT_NETWORK,
                "blind_canonical_login_failed",
                "Blind canonical connection or login failed",
            ) from None

        logger.info("Canonical Blind Ladder exact challenge mode ready")
        await _run_sequential_battles(
            client,
            prepared_deployment,
            process_config,
            public_prior_configuration,
            original_pokedex,
            original_move_json,
            battle_runner,
            integrity_checker,
        )
    except BlindCanonicalActivationError as error:
        activation_error = error
    finally:
        socket_close_failed = False
        owner_close_failed = False
        try:
            if client is not None:
                try:
                    await client.close()
                except _EXPECTED_NETWORK_ERRORS:
                    socket_close_failed = True
        finally:
            if prepared_deployment is not None:
                try:
                    prepared_deployment.close()
                except BlindPoolValidationError:
                    owner_close_failed = True
        if activation_error is None and socket_close_failed:
            activation_error = BlindCanonicalActivationError(
                BlindCanonicalErrorCategory.TRANSIENT_NETWORK,
                "blind_canonical_socket_close_failed",
                "Blind canonical connection cleanup failed",
            )
        if owner_close_failed:
            activation_error = BlindCanonicalActivationError(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Blind canonical deployment ownership could not be released",
            )
    if activation_error is not None:
        raise activation_error from None
