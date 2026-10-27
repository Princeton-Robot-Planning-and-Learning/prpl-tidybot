# Sprout robot-side experiment

This package contains the Sprout-specific robot runtime. No private Sprout model
assets belong in this repository.

- `live_runtime.py`, `live_perception.py`, `live_kinematics.py`: supervised execution, fresh camera/state reads, FK/IK and geometry guards.
- `sequence.py`, `compliant_backend.py`, `compliant_arm.py`: predefined compliant grasp/lift/lower/release cycle.
- `body_obstacle.py`, `base_planning.py`, `base_motion.py`: body keep-out, base-pose/route planning and guarded base transit.
- `rehearsal.py`: MuJoCo preview; `review.py`, `remote_execute.py`: approval and stop transport.
- `read_robot_state.py`, `shutdown_arm.py`: feedback capture and supervised retraction.
- Workstation orchestration, configuration examples and operator docs live in the sibling learning-forceful-manipulation worktree's `real_world/` directory.
- Shared `marker_detector/`, `visual_servo/` and `third_party/` code stays in its existing package location; moving those changes into this experiment would break or duplicate shared hardware interfaces.

No module should activate hardware merely by being imported. Use the documented
Mac workflow for preview and explicit execution approval.
