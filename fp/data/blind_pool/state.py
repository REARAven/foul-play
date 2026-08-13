"""Strict parsing and atomic persistence of Blind Ladder bag state."""

from __future__ import annotations

import errno
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from .errors import BlindPoolValidationError
from .fingerprint import compute_registry_fingerprint
from .models import (
    BlindPoolBagState,
    BlindPoolRegistry,
    BlindPoolReservation,
    BlindPoolStateConfig,
)


STATE_SCHEMA_VERSION = 1
RESERVATION_PHASE = "reserved"

_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "registry_fingerprint",
        "cycle_number",
        "cycle_order",
        "next_index",
        "last_consumed_id",
        "reservation",
    }
)
_RESERVATION_FIELDS = frozenset(
    {"reservation_id", "team_id", "cycle_number", "position", "phase"}
)
_TEAM_ID_PATTERN = re.compile(r"^BL-[0-9]{3,}-v[1-9][0-9]*$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RESERVATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class _DuplicateJsonFieldError(ValueError):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonFieldError
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError


def _validate_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if expected - value.keys():
        _fail(
            "state_missing_required_field",
            "{} is missing required fields".format(context),
        )
    if value.keys() - expected:
        _fail("state_unexpected_field", "{} contains unexpected fields".format(context))


def _valid_team_id(value: Any) -> bool:
    return isinstance(value, str) and _TEAM_ID_PATTERN.fullmatch(value) is not None


def _validate_reservation(
    value: Any,
    *,
    cycle_number: int,
    cycle_order: tuple[str, ...],
    next_index: int,
    last_consumed_id: str | None,
) -> BlindPoolReservation | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("state_reservation_invalid", "Blind Ladder reservation must be an object")
    _validate_exact_fields(
        value,
        _RESERVATION_FIELDS,
        context="Blind Ladder reservation",
    )
    reservation_id = value["reservation_id"]
    if (
        not isinstance(reservation_id, str)
        or _RESERVATION_ID_PATTERN.fullmatch(reservation_id) is None
    ):
        _fail("reservation_id_invalid", "Blind Ladder reservation ID is malformed")
    team_id = value["team_id"]
    if not _valid_team_id(team_id):
        _fail(
            "state_reservation_invalid",
            "Blind Ladder reservation team ID is invalid",
        )
    if value["phase"] != RESERVATION_PHASE:
        _fail(
            "reservation_phase_invalid",
            "Blind Ladder reservation phase is unsupported",
        )
    if next_index == len(cycle_order):
        _fail(
            "state_reservation_invalid",
            "A completed cycle cannot contain a reservation",
        )
    if type(value["cycle_number"]) is not int or value["cycle_number"] != cycle_number:
        _fail(
            "state_reservation_invalid",
            "Blind Ladder reservation cycle is inconsistent",
        )
    if type(value["position"]) is not int or value["position"] != next_index:
        _fail(
            "state_reservation_invalid",
            "Blind Ladder reservation position is inconsistent",
        )
    if team_id != cycle_order[next_index]:
        _fail(
            "state_reservation_invalid",
            "Blind Ladder reservation team is inconsistent",
        )
    if team_id == last_consumed_id:
        _fail(
            "state_reservation_invalid",
            "Blind Ladder reservation repeats the consumed team",
        )
    return BlindPoolReservation(
        reservation_id=reservation_id,
        team_id=team_id,
        cycle_number=cycle_number,
        position=next_index,
        phase=RESERVATION_PHASE,
    )


def validate_blind_pool_bag_state(
    document: Any,
    registry: BlindPoolRegistry,
) -> BlindPoolBagState:
    """Validate decoded state against current validated registry semantics."""

    if not isinstance(registry, BlindPoolRegistry):
        _fail("registry_type_invalid", "Blind Ladder registry has an invalid type")
    if not isinstance(document, dict):
        _fail("state_document_invalid", "Blind Ladder state must be an object")
    _validate_exact_fields(document, _TOP_LEVEL_FIELDS, context="Blind Ladder state")

    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != STATE_SCHEMA_VERSION:
        _fail(
            "state_schema_unsupported",
            "Blind Ladder state schema version is unsupported",
        )
    fingerprint = document["registry_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None
    ):
        _fail(
            "state_fingerprint_invalid",
            "Blind Ladder registry fingerprint is malformed",
        )
    if fingerprint != compute_registry_fingerprint(registry):
        _fail(
            "registry_fingerprint_mismatch",
            "Blind Ladder state does not match the registry",
        )

    cycle_number = document["cycle_number"]
    if type(cycle_number) is not int or cycle_number < 1:
        _fail("state_cycle_invalid", "Blind Ladder cycle number must be positive")
    order_value = document["cycle_order"]
    if not isinstance(order_value, list) or not all(
        _valid_team_id(item) for item in order_value
    ):
        _fail(
            "state_cycle_order_invalid",
            "Blind Ladder cycle order contains invalid IDs",
        )
    cycle_order = tuple(order_value)
    if len(set(cycle_order)) != len(cycle_order):
        _fail(
            "state_cycle_order_duplicate",
            "Blind Ladder cycle order contains duplicate IDs",
        )
    active_ids = {entry.team_id for entry in registry.active_entries}
    if set(cycle_order) != active_ids or len(cycle_order) != len(active_ids):
        _fail(
            "state_cycle_order_mismatch",
            "Blind Ladder cycle order does not match active IDs",
        )

    next_index = document["next_index"]
    if type(next_index) is not int or not 0 <= next_index <= len(cycle_order):
        _fail(
            "state_next_index_invalid",
            "Blind Ladder next position is outside the cycle",
        )
    last_consumed_id = document["last_consumed_id"]
    if last_consumed_id is not None and (
        not _valid_team_id(last_consumed_id) or last_consumed_id not in active_ids
    ):
        _fail("state_last_consumed_invalid", "Blind Ladder last consumed ID is invalid")
    if next_index > 0 and last_consumed_id != cycle_order[next_index - 1]:
        _fail(
            "state_last_consumed_invalid",
            "Blind Ladder last consumed ID is inconsistent",
        )
    if cycle_number == 1 and next_index == 0 and last_consumed_id is not None:
        _fail(
            "state_last_consumed_invalid",
            "Initial Blind Ladder state cannot have a consumed ID",
        )
    if cycle_number > 1 and last_consumed_id is None:
        _fail(
            "state_last_consumed_invalid",
            "Later Blind Ladder cycles require a consumed ID",
        )
    if (
        cycle_number > 1
        and next_index == 0
        and cycle_order[0] == last_consumed_id
    ):
        _fail(
            "state_cycle_boundary_repeat",
            "Blind Ladder state repeats a team across the cycle boundary",
        )

    reservation = _validate_reservation(
        document["reservation"],
        cycle_number=cycle_number,
        cycle_order=cycle_order,
        next_index=next_index,
        last_consumed_id=last_consumed_id,
    )
    return BlindPoolBagState(
        schema_version=STATE_SCHEMA_VERSION,
        registry_fingerprint=fingerprint,
        cycle_number=cycle_number,
        cycle_order=cycle_order,
        next_index=next_index,
        last_consumed_id=last_consumed_id,
        reservation=reservation,
    )


def load_blind_pool_bag_state(
    config: BlindPoolStateConfig,
    registry: BlindPoolRegistry,
) -> BlindPoolBagState:
    """Read one state file; callers must hold its transaction lock."""

    try:
        raw = config.state_path.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        _fail("state_not_initialized", "Blind Ladder state has not been initialized")
    except UnicodeDecodeError:
        _fail("state_encoding_invalid", "Blind Ladder state is not valid UTF-8")
    except OSError:
        _fail("state_file_unreadable", "Blind Ladder state could not be read")
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJsonFieldError:
        _fail(
            "state_duplicate_field",
            "Blind Ladder state JSON contains a duplicate field",
        )
    except (json.JSONDecodeError, ValueError):
        _fail("state_json_invalid", "Blind Ladder state contains malformed JSON")
    return validate_blind_pool_bag_state(document, registry)


def _state_document(state: BlindPoolBagState) -> dict[str, Any]:
    reservation = state.reservation
    return {
        "schema_version": state.schema_version,
        "registry_fingerprint": state.registry_fingerprint,
        "cycle_number": state.cycle_number,
        "cycle_order": list(state.cycle_order),
        "next_index": state.next_index,
        "last_consumed_id": state.last_consumed_id,
        "reservation": None
        if reservation is None
        else {
            "reservation_id": reservation.reservation_id,
            "team_id": reservation.team_id,
            "cycle_number": reservation.cycle_number,
            "position": reservation.position,
            "phase": reservation.phase,
        },
    }


def _serialize_state(state: BlindPoolBagState) -> str:
    return json.dumps(
        _state_document(state),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def write_blind_pool_bag_state_atomic(
    config: BlindPoolStateConfig,
    state: BlindPoolBagState,
    registry: BlindPoolRegistry,
) -> None:
    """Replace state from a flushed unique sibling file without in-place writes."""

    try:
        validated_state = validate_blind_pool_bag_state(
            _state_document(state),
            registry,
        )
        payload = _serialize_state(validated_state)
    except (TypeError, ValueError, UnicodeError):
        _fail(
            "atomic_state_write_failed",
            "Blind Ladder state could not be serialized",
        )

    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".{}-".format(config.state_path.name),
            suffix=".tmp",
            dir=config.state_path.parent,
        )
        temporary_path = Path(raw_path)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            errors="strict",
            newline="\n",
        ) as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, config.state_path)
        temporary_path = None
        _fsync_directory(config.state_path.parent)
    except (OSError, UnicodeError):
        _fail("atomic_state_write_failed", "Blind Ladder state could not be persisted")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
