from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from fp.config import FoulPlayConfig, SaveReplay
from fp.run_battle import pokemon_battle


class TerminalProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_terminal(self, message):
        client = mock.AsyncMock()
        client.receive_message.return_value = message
        battle = SimpleNamespace(battle_tag="battle-gen9tugs-401", wait=False)
        handler = mock.Mock()
        with (
            mock.patch(
                "fp.run_battle.start_battle",
                new=mock.AsyncMock(return_value=battle),
            ),
            mock.patch.object(
                FoulPlayConfig,
                "save_replay",
                SaveReplay.never,
                create=True,
            ),
        ):
            winner = await pokemon_battle(
                client,
                "gen9tugs",
                [],
                terminal_result_handler=handler,
            )
        return winner, handler, client

    async def test_authoritative_win_and_forfeit_winner_notify_before_return(self):
        winner, handler, client = await self.run_terminal(
            ">battle-gen9tugs-401\n|win|Synthetic Player\n"
        )
        self.assertEqual("Synthetic Player", winner)
        handler.assert_called_once_with("Synthetic Player", tied=False)
        client.leave_battle.assert_awaited_once_with("battle-gen9tugs-401")

    async def test_authoritative_tie_notifies_without_winner(self):
        winner, handler, _client = await self.run_terminal(
            ">battle-gen9tugs-401\n|tie|\n"
        )
        self.assertIsNone(winner)
        handler.assert_called_once_with(None, tied=True)

    async def test_disconnect_alone_never_notifies_result_handler(self):
        client = mock.AsyncMock()
        client.receive_message.side_effect = ConnectionError("synthetic disconnect")
        battle = SimpleNamespace(battle_tag="battle-gen9tugs-401", wait=False)
        handler = mock.Mock()
        with mock.patch(
            "fp.run_battle.start_battle",
            new=mock.AsyncMock(return_value=battle),
        ), self.assertRaises(ConnectionError):
            await pokemon_battle(
                client,
                "gen9tugs",
                [],
                terminal_result_handler=handler,
            )
        handler.assert_not_called()

    async def test_room_deinit_alone_never_notifies_result_handler(self):
        client = mock.AsyncMock()
        client.receive_message.side_effect = (
            ">battle-gen9tugs-401\n|deinit|\n",
            ConnectionError("synthetic disconnect"),
        )
        battle = SimpleNamespace(battle_tag="battle-gen9tugs-401", wait=True)
        handler = mock.Mock()
        with (
            mock.patch(
                "fp.run_battle.start_battle",
                new=mock.AsyncMock(return_value=battle),
            ),
            mock.patch(
                "fp.run_battle.async_update_battle",
                new=mock.AsyncMock(return_value=False),
            ),
            self.assertRaises(ConnectionError),
        ):
            await pokemon_battle(
                client,
                "gen9tugs",
                [],
                terminal_result_handler=handler,
            )
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
