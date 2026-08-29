"""Tests for the OmegaConf resolvers the env yamls use."""

from omegaconf import OmegaConf

import prpl_tidybot.config_resolvers  # noqa: F401  # pylint: disable=unused-import


def test_len_and_pluck_resolvers():
    """`${len:...}` counts a list and `${pluck:field,...}` extracts a field from each
    mapping in it, so a per-object env derives its counts from one list."""
    cfg = OmegaConf.create(
        {
            "cylinders": [{"height": 0.233, "r": 1}, {"height": 0.12, "r": 2}],
            "n": "${len:${cylinders}}",
            "heights": "${pluck:height,${cylinders}}",
        }
    )
    assert cfg.n == 2
    assert list(cfg.heights) == [0.233, 0.12]
