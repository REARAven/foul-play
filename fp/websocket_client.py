import asyncio
import ipaddress
import json
import re
import time
from urllib.parse import urlsplit

import logging
import requests
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class LoginError(Exception):
    pass


class LocalLoginConfigurationError(ValueError):
    pass


class SaveReplayError(Exception):
    pass


def validate_loopback_websocket_uri(address):
    try:
        parsed = urlsplit(address)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise LocalLoginConfigurationError(
            "--local-no-security-login requires a valid websocket URI"
        ) from error

    if parsed.scheme not in ("ws", "wss") or parsed.hostname is None:
        raise LocalLoginConfigurationError(
            "--local-no-security-login requires a valid ws:// or wss:// URI"
        )
    if parsed.username is not None or parsed.password is not None:
        raise LocalLoginConfigurationError(
            "--local-no-security-login does not allow URI credentials"
        )
    if port is None and not parsed.netloc:
        raise LocalLoginConfigurationError(
            "--local-no-security-login requires an explicit loopback destination"
        )

    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        return hostname
    try:
        address_ip = ipaddress.ip_address(hostname)
    except ValueError as error:
        raise LocalLoginConfigurationError(
            "--local-no-security-login requires localhost or a loopback IP address"
        ) from error
    if not address_ip.is_loopback:
        raise LocalLoginConfigurationError(
            "--local-no-security-login requires a loopback IP address"
        )
    return hostname


def _redact_received_message(message):
    redacted_lines = []
    for line in message.splitlines():
        if line.startswith("|challstr|"):
            line = "|challstr|<redacted>"
        elif line.startswith("|nametaken|"):
            line = "|nametaken|<redacted>"
        redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _to_id(value):
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class PSWebsocketClient:
    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    local_no_security_login = False
    last_message = None
    last_challenge_time = 0

    @classmethod
    async def create(
        cls, username, password, address, local_no_security_login=False
    ):
        if local_no_security_login:
            if password is not None:
                raise LocalLoginConfigurationError(
                    "Local no-security login cannot be combined with a password"
                )
            hostname = validate_loopback_websocket_uri(address)
            logger.info(
                "Login mode: local no-security (host={})".format(hostname)
            )
        else:
            logger.info("Login mode: public assertion")
        self = PSWebsocketClient()
        self.username = username
        self.password = password
        self.address = address
        self.local_no_security_login = local_no_security_login
        self.websocket = await websockets.connect(self.address)
        self.login_uri = (
            "https://play.pokemonshowdown.com/api/login"
            if password
            else "https://play.pokemonshowdown.com/action.php?"
        )
        return self

    async def join_room(self, room_name):
        message = "/join {}".format(room_name)
        await self.send_message("", [message])
        logger.debug("Joined room '{}'".format(room_name))

    async def receive_message(self):
        message = await self.websocket.recv()
        logger.debug(
            "Received message from websocket: {}".format(
                _redact_received_message(message)
            )
        )
        return message

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list)
        is_authentication = any(
            item.startswith("/trn ") for item in message_list
        )
        if is_authentication:
            logger.debug("Sending authentication message to websocket")
        else:
            logger.debug("Sending message to websocket: {}".format(message))
        await self.websocket.send(message)
        self.last_message = None if is_authentication else message

    async def avatar(self, avatar):
        await self.send_message("", ["/avatar {}".format(avatar)])
        await self.send_message("", ["/cmd userdetails {}".format(self.username)])
        while True:
            # Wait for the query response and check the avatar
            # |queryresponse|QUERYTYPE|JSON
            msg = await self.receive_message()
            msg_split = msg.split("|")
            if msg_split[1] == "queryresponse":
                user_details = json.loads(msg_split[3])
                if user_details["avatar"] == avatar:
                    logger.info("Avatar set to {}".format(avatar))
                else:
                    logger.warning(
                        "Could not set avatar to {}, avatar is {}".format(
                            avatar, user_details["avatar"]
                        )
                    )
                break

    async def close(self):
        await self.websocket.close()

    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            if split_message[1] == "challstr":
                return split_message[2], split_message[3]

    async def wait_for_challstr(self):
        while True:
            message = await self.receive_message()
            if any(line.startswith("|challstr|") for line in message.splitlines()):
                return

    async def local_no_security_login_and_confirm(self):
        await self.wait_for_challstr()
        await self.send_message("", ["/trn " + self.username + ",0,"])
        try:
            while True:
                message = await asyncio.wait_for(self.receive_message(), timeout=10)
                for line in message.splitlines():
                    if line.startswith("|nametaken|"):
                        logger.error("Local username claim was rejected")
                        raise LoginError("Local username claim was rejected")
                    if not line.startswith("|updateuser|"):
                        continue
                    fields = line.split("|", 4)
                    if (
                        len(fields) >= 4
                        and fields[3] == "1"
                        and _to_id(fields[2]) == _to_id(self.username)
                    ):
                        logger.info("Local username claim succeeded")
                        return self.username
        except TimeoutError as error:
            logger.error("Local username claim confirmation timed out")
            raise LoginError(
                "Local username claim confirmation timed out"
            ) from error
        except ConnectionClosed as error:
            logger.error("Connection closed before local username confirmation")
            raise LoginError(
                "Connection closed before local username confirmation"
            ) from error

    async def login(self):
        if self.local_no_security_login:
            logger.info("Logging in using local no-security mode...")
            return await self.local_no_security_login_and_confirm()

        logger.info("Logging in using public assertion mode...")
        client_id, challstr = await self.get_id_and_challstr()

        guest_login = self.password is None

        if guest_login:
            response = requests.post(
                self.login_uri,
                data={
                    "act": "getassertion",
                    "userid": self.username,
                    "challstr": "|".join([client_id, challstr]),
                },
            )
        else:
            response = requests.post(
                self.login_uri,
                data={
                    "name": self.username,
                    "pass": self.password,
                    "challstr": "|".join([client_id, challstr]),
                },
            )

        if response.status_code != 200:
            logger.error(
                "Could not get assertion (HTTP status {})".format(
                    response.status_code
                )
            )
            raise LoginError("Could not get assertion")

        if guest_login:
            assertion = response.text
        else:
            response_json = json.loads(response.text[1:])
            if "actionsuccess" not in response_json:
                logger.error("Login unsuccessful: assertion was not issued")
                raise LoginError("Could not log in")
            assertion = response_json.get("assertion")

        message = ["/trn " + self.username + ",0," + assertion]
        logger.info("Successfully logged in")
        await self.send_message("", message)
        await asyncio.sleep(3)
        return self.username if guest_login else response_json["curuser"]["userid"]

    async def update_team(self, team):
        await self.send_message("", ["/utm {}".format(team)])

    async def challenge_user(self, user_to_challenge, battle_format):
        logger.info("Challenging {}...".format(user_to_challenge))
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message("", message)
        self.last_challenge_time = time.time()

    async def accept_challenge(self, battle_format, room_name):
        if room_name is not None:
            await self.join_room(room_name)

        logger.info("Waiting for a {} challenge".format(battle_format))
        username = None
        while username is None:
            msg = await self.receive_message()
            split_msg = msg.split("|")
            if (
                len(split_msg) == 9
                and split_msg[1] == "pm"
                and split_msg[3].strip().replace("!", "").replace("‽", "")
                == self.username
                and split_msg[4].startswith("/challenge")
                and split_msg[5] == battle_format
            ):
                username = split_msg[2].strip()

        message = ["/accept " + username]
        await self.send_message("", message)

    async def search_for_match(self, battle_format):
        logger.info("Searching for ranked {} match".format(battle_format))
        message = ["/search {}".format(battle_format)]
        await self.send_message("", message)

    async def leave_battle(self, battle_tag):
        message = ["/leave {}".format(battle_tag)]
        await self.send_message("", message)

        while True:
            msg = await self.receive_message()
            if battle_tag in msg and "deinit" in msg:
                return

    async def save_replay(self, battle_tag):
        message = ["/savereplay"]
        await self.send_message(battle_tag, message)
