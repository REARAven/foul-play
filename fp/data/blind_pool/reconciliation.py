"""Private-identity-bound handles for offline ``accept_sent`` recovery."""

from __future__ import annotations

from enum import Enum
import hashlib
import re

from .errors import BlindPoolValidationError
from .models import BlindPoolBagState
from .state import ACCEPT_SENT_PHASE


RECONCILIATION_CASE_DOMAIN = b"TUGS-BLIND-RECONCILIATION-v1"
RECONCILIATION_CASE_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BlindReconciliationDisposition(str, Enum):
    """The two deliberate operator decisions permitted for quarantine."""

    CONSUMED = "consumed"
    NOT_CONSUMED = "not-consumed"


def validate_reconciliation_case(value: object) -> str:
    """Require the complete, exact, non-normalized reconciliation handle."""

    if (
        not isinstance(value, str)
        or RECONCILIATION_CASE_PATTERN.fullmatch(value) is None
    ):
        raise BlindPoolValidationError(
            "reconciliation_case_invalid",
            "Blind Ladder reconciliation case is malformed",
        ) from None
    return value


def _add_framed(digest: object, label: bytes, value: bytes) -> None:
    """Add one unambiguously labeled, length-framed byte string."""

    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def derive_reconciliation_case(state: BlindPoolBagState) -> str:
    """Derive a safe handle for exactly one durable ``accept_sent`` incident."""

    if not isinstance(state, BlindPoolBagState):
        raise BlindPoolValidationError(
            "reconciliation_state_invalid",
            "Blind Ladder reconciliation state is invalid",
        ) from None
    if state.reservation is None:
        raise BlindPoolValidationError(
            "reconciliation_not_applicable",
            "Blind Ladder state has no accepted challenge to reconcile",
        ) from None
    current = state.reservation
    if current.phase != ACCEPT_SENT_PHASE:
        raise BlindPoolValidationError(
            "reconciliation_not_applicable",
            "Blind Ladder state has no accepted challenge to reconcile",
        ) from None
    if current.challenge_token is None:
        raise BlindPoolValidationError(
            "reconciliation_identity_missing",
            "Blind Ladder accepted challenge identity is unavailable",
        ) from None

    digest = hashlib.sha256()
    _add_framed(digest, b"domain", RECONCILIATION_CASE_DOMAIN)
    _add_framed(
        digest,
        b"registry_fingerprint",
        state.registry_fingerprint.encode("ascii"),
    )
    _add_framed(digest, b"reservation_id", current.reservation_id.encode("ascii"))
    _add_framed(digest, b"team_id", current.team_id.encode("ascii"))
    _add_framed(digest, b"cycle_number", str(current.cycle_number).encode("ascii"))
    _add_framed(digest, b"position", str(current.position).encode("ascii"))
    _add_framed(digest, b"phase", current.phase.encode("ascii"))
    _add_framed(
        digest,
        b"challenge_token",
        current.challenge_token.wire_value().encode("ascii"),
    )
    return digest.hexdigest()
