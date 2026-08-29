"""OmegaConf resolvers the env yamls rely on.

Registered on import; :mod:`prpl_tidybot.pipeline` imports this module so any
entry point that composes a config through it (the planner script, the
tests) has them.

* ``${len:${env.cylinders}}`` — length of a list, so a per-object env can
  derive its object count from the list of objects instead of repeating it.
* ``${pluck:height,${env.cylinders}}`` — the named field of every mapping in
  a list, e.g. the heights of all cylinders for the simulator's env config.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import ListConfig, OmegaConf


def _length(items: Sequence[Any]) -> int:
    return len(items)


def _pluck(field: str, items: Sequence[Mapping[str, Any]]) -> ListConfig:
    # A ListConfig rather than a plain list: OmegaConf cannot store a raw list
    # in the value node it resolves an interpolation into (it raises
    # "Value 'list' is not a supported primitive type" when a config that
    # interpolates this is resolved in place, as hydra.utils.instantiate does).
    return OmegaConf.create([item[field] for item in items])


def register_resolvers() -> None:
    """Register the resolvers (idempotent)."""
    OmegaConf.register_new_resolver("len", _length, replace=True)
    OmegaConf.register_new_resolver("pluck", _pluck, replace=True)


register_resolvers()
