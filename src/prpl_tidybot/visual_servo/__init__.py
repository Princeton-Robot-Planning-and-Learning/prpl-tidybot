"""Visual servoing for the last mile of a cylinder grasp.

The pieces are deliberately separable so each can be exercised on its own:

* :mod:`cylinder_edges` finds the two vertical edges of a cylinder in a
  wrist-camera image with plain OpenCV, and draws a debug overlay.
* :mod:`image_sources` provides wrist images: the Kinova wrist camera on the
  robot, a fixed sequence of frames for tests, or the kinder simulator's
  end-effector camera rendered from a saved state.
* :mod:`tool_frame` turns small end-effector translations in the tool frame
  ("2 cm to the left", "1 cm forward") into joint targets with the standalone
  pybullet Kinova model.

:class:`prpl_tidybot.real_sim.plan_executors.visual_servo.CylinderVisualServoGapExecutor`
combines them into a gap executor for the magic grasp.
"""
