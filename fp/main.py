import json
import logging
from copy import deepcopy

from fp.config import (
    BotModes,
    FoulPlayConfig,
    TEAM_SOURCE_BLIND_CANONICAL,
    TEAM_SOURCE_LEGACY,
    init_logging,
)

from fp.modes import battle_mode
from fp.teams import load_team, TeamListIterator
from fp.run_battle import pokemon_battle
from fp.websocket_client import PSWebsocketClient

from fp.data import all_move_json
from fp.data import pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors.runtime import (
    load_public_prior_runtime_configuration,
)
from fp.data.blind_pool.activation import run_blind_canonical_activation
from fp.data.blind_pool.startup import (
    BlindCanonicalActivationError,
)

logger = logging.getLogger(__name__)


def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical(
            "Move JSON changed!\nDumping modified version to `modified_moves.json`"
        )
        with open("modified_moves.json", "w") as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical(
            "Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`"
        )
        with open("modified_pokedex.json", "w") as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")


async def _create_websocket(websocket_factory):
    return await websocket_factory(
        FoulPlayConfig.username,
        FoulPlayConfig.password,
        FoulPlayConfig.websocket_uri,
        FoulPlayConfig.local_no_security_login,
    )


async def _initialize_connection(ps_websocket_client):
    FoulPlayConfig.user_id = await ps_websocket_client.login()
    if FoulPlayConfig.avatar is not None:
        await ps_websocket_client.avatar(FoulPlayConfig.avatar)


async def _run_legacy_battles(
    ps_websocket_client,
    public_prior_configuration,
    original_pokedex,
    original_move_json,
):
    team_iterator = (
        None
        if FoulPlayConfig.team_list is None
        else TeamListIterator(FoulPlayConfig.team_list)
    )
    battles_run = 0
    wins = 0
    losses = 0
    team_dict = None
    mode = battle_mode(FoulPlayConfig.format_spec.battle_type)
    while True:
        if mode.requires_team:
            team_name = (
                team_iterator.get_next_team()
                if team_iterator is not None
                else FoulPlayConfig.team_name
            )
            team_packed, team_dict, _ = load_team(team_name)
            await ps_websocket_client.update_team(team_packed)
        else:
            await ps_websocket_client.update_team("None")

        if FoulPlayConfig.bot_mode == BotModes.challenge_user:
            await ps_websocket_client.challenge_user(
                FoulPlayConfig.user_to_challenge,
                FoulPlayConfig.pokemon_format,
            )
        elif FoulPlayConfig.bot_mode == BotModes.accept_challenge:
            await ps_websocket_client.accept_challenge(
                FoulPlayConfig.pokemon_format, FoulPlayConfig.room_name
            )
        elif FoulPlayConfig.bot_mode == BotModes.search_ladder:
            await ps_websocket_client.search_for_match(FoulPlayConfig.pokemon_format)
        else:
            raise ValueError("Invalid Bot Mode: {}".format(FoulPlayConfig.bot_mode))

        winner = await pokemon_battle(
            ps_websocket_client,
            FoulPlayConfig.pokemon_format,
            team_dict,
            public_prior_configuration=public_prior_configuration,
        )
        if winner == FoulPlayConfig.username:
            wins += 1
            logger.info("Battle won with selected team")
        else:
            losses += 1
            logger.info("Battle lost with selected team")

        logger.info("W: {}\tL: {}".format(wins, losses))
        check_dictionaries_are_unmodified(original_pokedex, original_move_json)

        battles_run += 1
        if battles_run >= FoulPlayConfig.run_count:
            break
    await ps_websocket_client.close()


async def run_foul_play(*, websocket_factory=None, environ=None):
    public_prior_options = FoulPlayConfig.configure()
    init_logging(FoulPlayConfig.log_level, FoulPlayConfig.log_to_file)
    apply_mods(FoulPlayConfig.format_spec)
    public_prior_configuration = load_public_prior_runtime_configuration(
        public_prior_options,
        FoulPlayConfig.pokemon_format,
    )

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)
    websocket_factory = websocket_factory or PSWebsocketClient.create
    team_source = getattr(FoulPlayConfig, "team_source", TEAM_SOURCE_LEGACY)
    if team_source == TEAM_SOURCE_LEGACY:
        ps_websocket_client = await _create_websocket(websocket_factory)
        await _initialize_connection(ps_websocket_client)
        await _run_legacy_battles(
            ps_websocket_client,
            public_prior_configuration,
            original_pokedex,
            original_move_json,
        )
        return
    if team_source != TEAM_SOURCE_BLIND_CANONICAL:
        raise ValueError("Invalid team source")

    activation_error = None
    try:
        await run_blind_canonical_activation(
            FoulPlayConfig,
            public_prior_configuration,
            original_pokedex,
            original_move_json,
            websocket_factory=websocket_factory,
            environ=environ,
            battle_runner=pokemon_battle,
            integrity_checker=check_dictionaries_are_unmodified,
        )
    except BlindCanonicalActivationError as error:
        activation_error = error
    if activation_error is not None:
        logger.error(
            "Blind canonical startup failed [{}:{}]".format(
                activation_error.category.value,
                activation_error.code,
            )
        )
        raise SystemExit(1) from None
