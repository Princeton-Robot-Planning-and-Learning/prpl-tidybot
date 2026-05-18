"""Gymnasium env wrapping a kinder kinematic3d sim as the world.

The fake/real path goes `Interface -> RealTidyBotEnv (TidyBotObservation) ->
PrplLab3DPerceiver (ObjectCentricState) -> Agent`. In sim mode the kinder
env already produces an ObjectCentricState, so re-encoding it as a
TidyBotObservation just to decode it again would be lossy nonsense. This
wrapper exposes the kinder env's state directly to the Runner; pair it
with the pass-through perceiver and action grounder under
`prpl_tidybot.real_sim` to drive the existing kinder-state agent loop
without leaving sim.
"""

from typing import Any, SupportsFloat

import gymnasium
import kinder
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace


class PrplLab3DSimEnv(
    gymnasium.Env[ObjectCentricState, NDArray]  # type: ignore[type-arg]
):
    """A thin wrapper that devectorizes obs from the underlying kinder env.

    Defaults to `kinder/PrplLab3D-o1-v0`; override `env_id` for other
    variants. The wrapped env owns the kinder state; this class just
    forwards `reset` / `step` / `render` and unwraps the observation.
    """

    def __init__(self, env_id: str = "kinder/PrplLab3D-o1-v0") -> None:
        kinder.register_all_environments()
        self._env = gymnasium.make(env_id, render_mode="rgb_array")
        # kinder envs serialize ObjectCentricState through ObjectCentricBoxSpace;
        # we devectorize back to the structured state in reset/step.
        assert isinstance(self._env.observation_space, ObjectCentricBoxSpace)
        self._obs_space: ObjectCentricBoxSpace = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        super().reset(seed=seed)
        obs_vec, info = self._env.reset(seed=seed, options=options)
        return self._obs_space.devectorize(obs_vec), info

    def step(
        self, action: NDArray
    ) -> tuple[ObjectCentricState, SupportsFloat, bool, bool, dict[str, Any]]:
        obs_vec, reward, terminated, truncated, info = self._env.step(action)
        return (
            self._obs_space.devectorize(obs_vec),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
