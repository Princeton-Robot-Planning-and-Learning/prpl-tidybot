"""Where the visual servo gets its wrist images.

The plan executors only see perceived states, so an executor that needs
pixels holds its own :class:`ImageSource`. Three are provided:

* :class:`KinovaWristCameraSource` — the robot's wrist camera (RTSP through
  the GStreamer OpenCV build on the NUC). Opened lazily on first use so
  that instantiating a pipeline config off the robot does not try to
  connect.
* :class:`SequenceImageSource` — a fixed list of frames, for tests and for
  replaying saved images.
* :class:`KinderEECameraSource` — the kinder simulator's end-effector camera
  rendered from an object-centric state, with the arm joints overridden by
  a callable. Used to exercise the whole servo loop against real renders
  without a robot.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

import numpy as np
from prpl_utils.structs import Image
from relational_structs import ObjectCentricState

from prpl_tidybot.third_party.cameras import KinovaCamera


class ImageSource(Protocol):
    """Anything that can produce the current wrist image."""

    def get_image(self) -> Image | None:
        """Return the latest RGB frame, or None if none is available yet."""


class SequenceImageSource:
    """Replay a list of frames; the last frame repeats once the list is exhausted."""

    def __init__(self, frames: Sequence[Image]) -> None:
        if not frames:
            raise ValueError("SequenceImageSource needs at least one frame.")
        self._frames = [np.asarray(f) for f in frames]
        self._index = 0

    def get_image(self) -> Image | None:
        """The next frame in the sequence."""
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame


class KinovaWristCameraSource:
    """The Kinova wrist camera, opened on first :meth:`get_image`."""

    def __init__(self) -> None:
        self._camera: Any = None

    def get_image(self) -> Image | None:
        """The latest wrist frame, opening the stream on first use."""
        if self._camera is None:
            self._camera = KinovaCamera()  # type: ignore[no-untyped-call]
        image = self._camera.get_image()
        return None if image is None else np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        """Release the camera stream if it was opened."""
        if self._camera is not None:
            self._camera.close()
            self._camera = None


class KinderEECameraSource:
    """Render the kinder end-effector camera for a state whose arm joints come from
    ``joints_fn`` (typically the executor's last commanded target)."""

    def __init__(
        self,
        env: Any,
        base_state: ObjectCentricState,
        joints_fn: Callable[[], Sequence[float]],
        robot_name: str = "robot",
    ) -> None:
        self._env = env
        self._base_state = base_state
        self._joints_fn = joints_fn
        self._robot_name = robot_name

    def get_image(self) -> Image | None:
        """Render the wrist camera at the base state with the current joints."""
        state = self._base_state.copy()
        robot = state.get_object_from_name(self._robot_name)
        for index, value in enumerate(self._joints_fn()):
            state.set(robot, f"joint_{index + 1}", float(value))
        self._env.set_state(state)
        return np.asarray(self._env.render_ee_camera(), dtype=np.uint8)
