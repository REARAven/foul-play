"""Explicit local-file startup configuration for public opponent priors."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from fp.battle.public_prior_context import (
    PublicPriorFallback,
    PublicPriorSearchContext,
)
from fp.battle.helpers import normalize_name
from fp.constants import BattleType
from fp.data.public_priors.errors import (
    PublicPriorRegistryError,
    PublicPriorValidationError,
)
from fp.data.public_priors.loader import load_public_prior
from fp.data.public_priors.models import (
    PublicPriorIdentity,
    PublicPriorRegistry,
)
from fp.format_spec import FormatSpec


logger = logging.getLogger(__name__)
_URL_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class PublicPriorConfigurationError(Exception):
    """A command-line public-prior configuration cannot be used safely."""


@dataclass(frozen=True)
class PublicPriorStartupOptions:
    """Immutable raw CLI choices retained only until startup loading."""

    file_paths: tuple[str, ...]
    fallback_policy: PublicPriorFallback

    def __post_init__(self) -> None:
        paths = tuple(self.file_paths)
        if not paths or not all(isinstance(path, str) and path for path in paths):
            raise ValueError("file_paths must contain explicit nonempty local paths")
        if not isinstance(self.fallback_policy, PublicPriorFallback):
            raise TypeError("fallback_policy must be PublicPriorFallback.GENERIC or NONE")
        object.__setattr__(self, "file_paths", paths)


@dataclass(frozen=True)
class PublicPriorRuntimeConfiguration:
    """Immutable process-level public data and per-battle context factory."""

    registry: PublicPriorRegistry
    selected_identities: tuple[PublicPriorIdentity, ...]
    fallback_policy: PublicPriorFallback
    format_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.registry, PublicPriorRegistry):
            raise TypeError("registry must be a PublicPriorRegistry")
        identities = tuple(self.selected_identities)
        if not identities or not all(
            isinstance(identity, PublicPriorIdentity) for identity in identities
        ):
            raise TypeError("selected_identities must contain public identities")
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate selected public-prior identity")
        if not isinstance(self.fallback_policy, PublicPriorFallback):
            raise TypeError("fallback_policy must be PublicPriorFallback.GENERIC or NONE")
        if (
            not isinstance(self.format_id, str)
            or not self.format_id
            or normalize_name(self.format_id) != self.format_id
        ):
            raise ValueError("format_id must be a canonical normalized ID")
        if any(identity.format_id != self.format_id for identity in identities):
            raise ValueError("all selected public-prior identities must match format_id")
        object.__setattr__(self, "selected_identities", identities)

    def create_battle_context(self, battle_format: str) -> PublicPriorSearchContext:
        """Return a fresh context while sharing only immutable public records."""

        if battle_format != self.format_id:
            raise PublicPriorConfigurationError(
                "battle format {!r} does not match configured public-prior format {!r}".format(
                    battle_format, self.format_id
                )
            )
        return PublicPriorSearchContext(
            registry=self.registry,
            selected_identities=self.selected_identities,
            fallback_policy=self.fallback_policy,
            format_id=self.format_id,
        )


def _validation_details(error: PublicPriorValidationError) -> str:
    """Render useful issue paths/messages without dumping invalid source values."""

    return "; ".join(
        "{}: {}".format(issue.path, issue.explanation) for issue in error.issues
    )


def load_public_prior_runtime_configuration(
    options: PublicPriorStartupOptions | None,
    configured_format: str,
) -> PublicPriorRuntimeConfiguration | None:
    """Load explicit files once, in order, after mechanics overlays are active."""

    if options is None:
        return None
    if not isinstance(options, PublicPriorStartupOptions):
        raise TypeError("options must be PublicPriorStartupOptions or None")
    if (
        not isinstance(configured_format, str)
        or not configured_format
        or normalize_name(configured_format) != configured_format
    ):
        raise PublicPriorConfigurationError(
            "--pokemon-format must be a canonical normalized format ID"
        )
    if FormatSpec.from_format_string(configured_format).battle_type is not (
        BattleType.STANDARD_BATTLE
    ):
        raise PublicPriorConfigurationError(
            "--public-prior-file is supported only for standard-battle formats"
        )

    datasets = []
    selected_identities = []
    identity_paths: dict[PublicPriorIdentity, str] = {}
    for raw_path in options.file_paths:
        if _URL_PREFIX.match(raw_path):
            raise PublicPriorConfigurationError(
                "--public-prior-file accepts local filesystem paths only: {!r}".format(
                    raw_path
                )
            )
        source = Path(raw_path)
        try:
            dataset = load_public_prior(source)
        except PublicPriorValidationError as error:
            raise PublicPriorConfigurationError(
                "--public-prior-file {!r} failed validation: {}".format(
                    raw_path, _validation_details(error)
                )
            ) from error

        if dataset.identity.format_id != configured_format:
            raise PublicPriorConfigurationError(
                "--public-prior-file {!r} has format_id {!r}; expected --pokemon-format {!r}".format(
                    raw_path, dataset.identity.format_id, configured_format
                )
            )
        previous_path = identity_paths.get(dataset.identity)
        if previous_path is not None:
            raise PublicPriorConfigurationError(
                "--public-prior-file {!r} duplicates public dataset identity {} already loaded from {!r}".format(
                    raw_path, dataset.identity, previous_path
                )
            )
        identity_paths[dataset.identity] = raw_path
        datasets.append(dataset)
        selected_identities.append(dataset.identity)

    try:
        registry = PublicPriorRegistry(tuple(datasets))
    except PublicPriorRegistryError as error:
        raise PublicPriorConfigurationError(
            "could not construct immutable public-prior registry: {}".format(error)
        ) from error

    configuration = PublicPriorRuntimeConfiguration(
        registry=registry,
        selected_identities=tuple(selected_identities),
        fallback_policy=options.fallback_policy,
        format_id=configured_format,
    )
    for position, identity in enumerate(configuration.selected_identities, start=1):
        logger.info(
            "Public prior precedence {}: dataset_id={} dataset_version={} format_id={} fallback={}".format(
                position,
                identity.dataset_id,
                identity.dataset_version,
                identity.format_id,
                configuration.fallback_policy.value,
            )
        )
    return configuration
