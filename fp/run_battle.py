from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fp import constants
from fp.config import FoulPlayConfig, SaveReplay
from fp.battle.protocol import async_update_battle
from fp.format_spec import FormatSpec
from fp.modes import battle_mode
from fp.modes.base import async_pick_move

if TYPE_CHECKING:
    from fp.data.public_priors.runtime import PublicPriorRuntimeConfiguration

logger = logging.getLogger(__name__)


def battle_is_finished(battle_tag, msg):
    return (
        msg.startswith(">{}".format(battle_tag))
        and (constants.WIN_STRING in msg or constants.TIE_STRING in msg)
        and constants.CHAT_STRING not in msg
    )


async def enable_battle_timer_once(ps_websocket_client, battle_tag):
    """Enable the timer once per room when automatic timer control is enabled."""

    if not FoulPlayConfig.battle_timer:
        return

    timer_rooms = vars(ps_websocket_client).setdefault(
        "_foul_play_timer_rooms", set()
    )
    if battle_tag in timer_rooms:
        return

    await ps_websocket_client.send_message(battle_tag, ["/timer on"])
    timer_rooms.add(battle_tag)


async def start_battle(
    ps_websocket_client,
    pokemon_battle_type,
    team_dict,
    *,
    public_prior_configuration: PublicPriorRuntimeConfiguration | None = None,
):
    format_spec = FormatSpec.from_format_string(pokemon_battle_type)
    mode = battle_mode(format_spec.battle_type)
    start_kwargs = {}
    if public_prior_configuration is not None:
        start_kwargs["public_prior_context"] = (
            public_prior_configuration.create_battle_context(pokemon_battle_type)
        )
    battle = await mode.start_battle(
        ps_websocket_client,
        pokemon_battle_type,
        team_dict,
        **start_kwargs,
    )

    await ps_websocket_client.send_message(battle.battle_tag, ["hf"])
    await enable_battle_timer_once(ps_websocket_client, battle.battle_tag)

    return battle


async def pokemon_battle(
    ps_websocket_client,
    pokemon_battle_type,
    team_dict,
    *,
    public_prior_configuration: PublicPriorRuntimeConfiguration | None = None,
):
    battle = await start_battle(
        ps_websocket_client,
        pokemon_battle_type,
        team_dict,
        public_prior_configuration=public_prior_configuration,
    )
    while True:
        msg = await ps_websocket_client.receive_message()
        if battle_is_finished(battle.battle_tag, msg):
            winner = (
                msg.split(constants.WIN_STRING)[-1].split("\n")[0].strip()
                if constants.WIN_STRING in msg
                else None
            )
            logger.info("Winner: {}".format(winner))
            await ps_websocket_client.send_message(battle.battle_tag, ["gg"])
            if (
                FoulPlayConfig.save_replay == SaveReplay.always
                or (
                    FoulPlayConfig.save_replay == SaveReplay.on_loss
                    and winner != FoulPlayConfig.username
                )
                or (
                    FoulPlayConfig.save_replay == SaveReplay.on_win
                    and winner == FoulPlayConfig.username
                )
            ):
                await ps_websocket_client.save_replay(battle.battle_tag)
            await ps_websocket_client.leave_battle(battle.battle_tag)
            return winner
        else:
            action_required = await async_update_battle(battle, msg)
            if action_required and not battle.wait:
                best_move = await async_pick_move(battle)
                await ps_websocket_client.send_message(battle.battle_tag, best_move)
