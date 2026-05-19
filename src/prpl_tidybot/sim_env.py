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


class KinderSimEnv(
    gymnasium.Env[ObjectCentricState, NDArray]  # type: ignore[type-arg]
):
    """Thin wrapper that devectorizes obs from any kinematic3d kinder env.

    Pass the gymnasium id (e.g. `"kinder/BaseMotion3D-v0"`). The wrapped
    env owns the kinder state; this class just forwards `reset` / `step`
    / `render` and exchanges the raw vectorized obs for an
    `ObjectCentricState`.
    """

    def __init__(self, env_id: str, **make_kwargs: Any) -> None:
        kinder.register_all_environments()
        self._env = gymnasium.make(env_id, render_mode="rgb_array", **make_kwargs)
        # kinder envs serialize ObjectCentricState through
        # ObjectCentricBoxSpace; we devectorize back to the structured
        # state in reset/step.
        assert isinstance(self._env.observation_space, ObjectCentricBoxSpace)
        self._obs_space: ObjectCentricBoxSpace = self._env.observation_space
        # Expose the inner env's spaces so callers that introspect
        # `env.observation_space` / `env.action_space` see something
        # meaningful (gymnasium.Env's class-level defaults would otherwise
        # be unset). The observation_space is the *vectorized* box space
        # even though `step` returns a devectorized ObjectCentricState —
        # the env-model factory in kinder-baselines reads box dimensions
        # off this space.
        self.observation_space = self._env.observation_space  # type: ignore[assignment]
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

    def set_state(self, state: ObjectCentricState) -> None:
        """Teleport the inner kinder env to `state`.

        Requires the env to be constructed with `allow_state_access=True`
        (see the sim pipeline yaml). Used by the recording layer to
        render the agent's perceived state on a shadow sim.
        """
        self._env.unwrapped._object_centric_env.set_state(  # type: ignore[attr-defined]  # pylint: disable=protected-access
            state
        )

    def close(self) -> None:
        self._env.close()
