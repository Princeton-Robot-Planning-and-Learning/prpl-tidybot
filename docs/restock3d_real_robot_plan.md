# Restock3D on the real TidyBot: integration plan

Goal: a video of a SPECTRE test-time plan (alphatamp, `experiments/spectre/restock3d_deploy`)
executed on the real TidyBot. Minimum: one successful real run. Ideal: the usual
real-to-sim-to-real loop, where the lab scene is perceived, planned in alphatamp, and the
plan is executed by the prpl-tidybot stack.

## 1. Where things stand (verified 2026-08-31)

**alphatamp deploy kit (branch `tidybot-deployment`).** `deploy.py` works end to end on the
bundled `scenes/demo6`: plan found on attempt 2 after 108 s, `plan_level_a.json`,
`plan_level_b.{json,npz}` and `plan.mp4` (72 s, 640x480) written. The Level-B trajectory is
what matters for execution:

- 2869 timesteps, arrays `base (T,3)`, `joints (T,7)`, `gripper (T,)`, `actions (T-1,11)`,
  per-object poses. Robot is kinder's `tidybot-kinova`, the same model prpl-tidybot's kinder
  envs use. World frame: origin at the robot's start pose, +x lateral, +y toward the shelf.
- Every step is base-only or arm-only. Zero mixed steps; 24 alternating segments (one base
  move and one arm segment per operator). This is the segment structure
  `Kinematic3DPlanExecutor` already expects, so its base/arm XOR check passes as-is.
- Actions are the kinder 11-d delta layout `[dx, dy, dtheta, dj1..dj7, gripper]` with
  gripper ±1, the same convention `CarrotArmMotion3DPlanExecutor._gripper_target` decodes.
  Step sizes: ≤0.096 m base, ≤0.13 rad joints. Gripper state crosses the 0.01 threshold
  exactly 6 times each way (the ±1 command repeats for 2-3 steps per event).
- Level-A waypoints match the Level-B states at their `timestep` fields exactly.
- Base standoff for places is 0.85 m in front of the shelf, the same distance the
  cylinder-shelf env settled on (0.86).

**prpl-tidybot (branch `test-skippy-servo`).** The real pipeline is complete: ceiling ArUco
marker detector (perception laptop, port 6002) → `CylinderShelf3DPerceiver` → kinder
`BilevelPlanningAgent` → `Kinematic3DPlanExecutor` (pure-pursuit base + carrot arm) →
`RealTidyBotEnv` (base RPC on the NUC, arm RPC on port 50001, Robotiq gripper). Grasp
last-mile is handled either by `TeleopGapExecutor` or `CylinderVisualServoGapExecutor`. The
preview renders the plan in a shadow sim and refuses any plan whose base leaves
`preview.floor_bounds`. The README's claim that the arm and camera interfaces raise
`NotImplementedError` is stale; both are implemented.

**Frames.** prpl-tidybot's map frame is defined by the floor markers
(`conf/calibration/prpl.yaml`, floor tiles ±1.83 m). The shelf is staged at map
`(1.5, 1.5, 0.02)`. alphatamp's home frame has the shelf at `(0.4, 1.4)`, open front facing
−y. The two are related by one SE2 transform (section 3).

**Shelf.** Both sims model the same physical shelf: 0.60198 × 0.254 m boards, 0.0127 m thick.
kinder's `CylinderShelf3DEnvConfig` has four board slots at 0.2667 m pitch, so board
surfaces sit at z ≈ 0.026, 0.293, 0.560, 0.827. Restock3D places its bottom (tall) section
on a surface at z = 0.29, which is slot 1. The section gaps differ: Restock3D uses
0.27 (tall) / 0.22 (short) above that surface; the physical slots give 0.254 / 0.254. The
F3 height cutoffs in `feasibility_v3.py` (`TALL_CUTOFF=0.17`, `SHORT_CUTOFF=0.12`) are
pinned constants and do not follow the `sections.clearances` override in `scene.yaml`, so
the real shelf should be staged to the trained geometry rather than the sim adjusted to
the shelf. The cylinder-shelf work currently has slot 1 removed (for the tall Pringles
can); it goes back in for Restock3D.

**Toolchain split.** alphatamp requires Python ≥3.11 and pins `kindergarden==0.2.0` +
kinder-baselines `4c731dc`; prpl-tidybot requires Python <3.11 and pins a kindergarden git
commit + kinder-baselines `98a5ef6`. The installed prpl-tidybot kinder has no Restock3D env
(it lives in alphatamp's tree). The two stacks therefore stay in separate venvs and the
`plan_level_b.npz` file is the seam between them. This is what the deploy kit was designed
for and it is the right boundary.

## 2. Minimum goal: one successful real run by trajectory replay

The exported joint stream is replayed through prpl-tidybot's existing executors. No
planning happens on the robot side. Sections 2.1-2.6 describe the pieces; section 6 gives
the execution order, which front-loads everything that does not need the robot.

### 2.1 Physical staging (lab, ~1 hour)

1. Reinstall the shelf's slot-1 board. Measure every board surface height and the
   inside gap above slot 1 and slot 2. If the boards are peg-adjustable, set the gaps to
   0.27 and 0.22 (the trained geometry). If they are fixed at 0.254 / 0.254, keep them and
   constrain object heights as in step 3.
2. Confirm the shelf's yaw in the map frame (which map-frame direction its open front
   faces). The cylinder-shelf staging already fixes this; write it down as `shelf_yaw_map`.
3. Choose 6 objects (the trained count; 6 is the smallest stratum and plans fastest). The
   front grasp closes on the object's ±x faces with a ~0.09 m aperture, so:
   - width 0.04-0.07 m; cylinders and jars are best because their yaw does not matter and
     the floor marker gives only (x, y). Boxes must be set down axis-aligned to the shelf.
   - two "tall" objects with height in (0.12, 0.15] (sim routes them to the bottom section;
     0.15 keeps ≥0.10 m insertion headroom under a physical 0.254 gap), four "short" objects
     with height ≤ 0.12. The Skippy jar (r 0.035, h 0.125) is a ready-made tall.
   - packing: per section, Σwidth + 0.06·(n−1) + 0.08 ≤ 0.50.
4. Tape one target marker (ids 35-46, `conf/calibration/`) on the floor under each object,
   inside the staging band. In the home frame the band is x ∈ (−0.80, −0.20), y ∈ (0.60,
   1.20), centers ≥0.12 m apart; section 3 gives the map-frame equivalent.

### 2.2 Define the home frame (once)

Restock3D fixes the shelf at home-frame `(0.4, 1.4)`, so the home frame is defined from the
shelf, not from wherever the robot happens to be:

```
home_origin_map = shelf_map ⊕ R(shelf_yaw_map) · (−0.4, −1.4)
home_yaw_map    = shelf_yaw_map            (home +y points into the shelf's open front)
```

With the shelf at map `(1.5, 1.5)` and its front facing map −y, the home origin is map
`(1.1, 0.1)`, the object band is map x ∈ (0.3, 0.9), y ∈ (0.7, 1.3), and the planned base
poses (picks at home y ≈ 0.0-0.35, places at home x ≈ 0.2-0.55, y ≈ 0.55) land at map
x ≤ 1.65, y ∈ (0.1, 0.7). Everything is inside `preview.floor_bounds` `[-1.5, 1.7, -1.5, 1.5]`
with a few centimetres to spare on the +x side. Stage the robot at the home origin
facing +x (the sim's start pose is SE2 identity); the pure-pursuit tracker absorbs a few
centimetres of staging error because it projects the base onto the planned polyline.

### 2.3 Real-to-sim, version 0 (manual, ~half day)

A small script on the prpl-tidybot side, `scripts/export_restock_scene.py`:

- reads the marker detector once (`MarkerDetectorClient`, the same call
  `MarkerDetectorCylinderTargets` makes at reset): robot map pose + target (x, y) per
  marker id;
- takes a yaml table `{marker_id: {name, width, height, depth}}` (the analogue of the
  `cylinders` list in `kinematic-cylinder-shelf3d.yaml`);
- accepts a `--from-json <payload>` flag that substitutes a recorded marker-detector
  payload for the live query, so the whole script is testable away from the robot;
- transforms each target into the home frame with the transform above and writes
  alphatamp's `scene.yaml` (objects with `floor: [x, y]`, plus `shelf: {x, y}` computed from
  the measured shelf position so residual staging error goes into the sim rather than
  being ignored);
- prints the robot's home-frame pose so the operator can see how far off the start is.

Small shelf offsets are safe (the layout shifts rigidly); shelf yaw is not overridable, so
the robot's heading at staging is what must be right.

### 2.4 Plan (laptop, alphatamp venv, minutes)

```
python deploy.py --scene scenes/lab_run1 --render
```

Review `plan.mp4` (this is the sim half of the final video) and `plan_level_a.json`. If
`NO PLAN FOUND`, raise `--k-max`. Copy `outputs/lab_run1/plan_level_b.npz` to the NUC.

### 2.5 prpl-tidybot changes for replay (~1-2 days)

1. `src/prpl_tidybot/replay_agent.py`: `NpzPlanAgent(PlanningAgent)`. `plan()` returns
   `[(state_t, action_t)]` built from the npz: states via `create_state_from_dict(...)` with
   `pos_base_{x,y,rot}`, `joint_1..7`, `finger_state` (copy the robot block from
   `perceivers/kinematic3d.py:92-126`); actions are the exported 11-d deltas. Apply the
   home→map SE2 transform to `base` and rotate the `dx, dy` deltas by `home_yaw_map` before
   building pairs. Set `_planned_states` / `_planned_actions` so the preview keeps working.
   `plan()` is called once per episode, which is all the Runner needs.
2. `pipeline.py:122`: make the agent a Hydra `_target_` block (default
   `BilevelPlanningAgent`) so the replay yaml can select `NpzPlanAgent` with
   `plan_path=...`, `home_origin_map`, `home_yaw_map`.
3. `conf/env/restock3d-replay.yaml`: copy the real pipeline block of the cylinder-shelf
   yaml (same `RealInterface`, `PurePursuitBaseMotion3DPlanExecutor`,
   `CarrotArmMotion3DPlanExecutor`), with a perceiver that only reports the robot (a
   `Kinematic3DFixtureType` shelf and the objects as static boxes are enough for the
   preview). Raise `max_iter_total` / `max_iter_per_pair`: arm segments here are ~200-260
   waypoints (the cylinder env's are shorter), and each segment is reach + close + stow.
   Keep `arm_lookahead 0.25`, `advance_radius 0.15`, `gripper_dwell_ticks 20`;
   `gripper_close_position` is global, so pick it for the narrowest object or add a
   per-segment override keyed by Level-A `object`.
4. Shadow sim for the preview: `KinderSimEnv` cannot build Restock3D. Either run the
   preview with `CylinderShelf3D` as a stand-in (robot + shelf render; boxes missing) or
   skip the render and keep only `find_floor_violation`, which needs no sim. The
   alphatamp `plan.mp4` is the real preview.
5. Before the first base segment, settle the arm to `joints[0]` (the sim home
   configuration) with a `SettleGapExecutor`-style unplanned move; the first arm segment
   assumes it starts there.

### 2.6 Dry runs, in order

1. `mode=fake` with the npz: exercises the agent, segmentation, gripper decoding, preview,
   floor-bounds check. Must pass CI (`./run_ci_checks.sh`).
2. `mode=real` with `arm_interface` faked (as `base_motion3d` does): base-only replay on
   the empty floor. Checks the frame transform end to end: the base should stop 0.72 m
   short of each marker and 0.85 m in front of the shelf.
3. `mode=real`, full arm, no objects on the floor: the "air run". Watch reach depth at the
   grasp poses (the arm is near full extension at 0.72 m standoff) and the insertion into
   both shelf sections. E-stop in reach; this is where a URDF/mount mismatch would show.
4. Full run with objects. Expect the first grasps to miss by a few centimetres (the
   cylinder-shelf notes report base staging errors up to ~4 cm and a controller deadband of
   2-3 cm); that is what section 4 addresses. Re-run until one clean success.
5. Video: `record.video=true` records the ceiling camera through `CeilingCameraRenderer`.
   Compose it with alphatamp's `plan.mp4` offline (the recorder's side-by-side needs a
   shadow sim that can render Restock3D, which prpl-tidybot does not have).

## 3. Ideal goal: closed real-to-sim-to-real loop

Everything in section 2 is already most of the loop; what remains is automation and grasp
robustness.

1. **One-command orchestration.** A laptop-side script: perceive (section 2.3) → write
   `scene.yaml` → run `deploy.py` in the alphatamp venv → scp the npz to the NUC → launch
   `run_planner.py env=restock3d-replay mode=real` there. The marker detector is reachable
   from the laptop (`lab.marker_detector_host`), so perception does not need the NUC.
2. **Grasp holes (round 2).** Convert each pick's approach+close into a magic gap: emit the
   Level-B pairs up to the pre-grasp state (0.12 m back along the approach axis; the
   Level-A `timestep` minus the approach steps), then one `SkillCall` whose
   `predicted_state` is the post-stow state from the export, then continue. The existing
   `TeleopGapExecutor` handles the gap with a human on the gamepad and settles to the
   predicted joints. This turns a 5-cm grasp miss into a 10-second teleop nudge and keeps
   the rest of the plan autonomous.
3. **Front-grasp visual servo (round 3).** `CylinderVisualServoGapExecutor` is written for
   the side grasp: it aligns the cylinder laterally in the wrist camera, then approaches
   along the tool axis with a parallax range estimate. The front grasp differs only in
   the pre-grasp standoff (0.12 vs 0.10 m) and the gripper's approach direction, so
   most of it transfers; boxes need an edge detector that tolerates flat faces (the
   current one fits cylinder silhouettes and holds widths to the radius). Do this only
   after round 2 succeeds.
4. **Perception of sizes.** Today sizes are typed in per marker. That is fine for a
   proof of concept; a depth-camera box fit would replace the table later.
5. **SPECTRE as a `PlanningAgent` inside the Runner (later).** The cleanest integration
   is a `SpectreRestock3DAgent` that takes the perceiver's state and runs the deploy
   loop in-process, which would also allow replanning after a failed grasp. It is
   blocked by the Python and kinder pin skew (alphatamp ≥3.11, prpl-tidybot <3.11). Two
   ways out: lift prpl-tidybot to 3.11 (the `<3.11` cap comes from the kortex wheel;
   check whether `kortex_wheels/` has a 3.11 build) or vendor the Restock3D env +
   ranker into a package both can import. Not needed for the video.

## 4. Risks and mitigations

| Risk | Where it bites | Mitigation |
|---|---|---|
| Shelf gaps 0.254 vs trained 0.27/0.22 | tall-section insertion headroom | cap tall objects at 0.15 m, or adjust pegs |
| Home-frame yaw error | every place misses laterally | define yaw from the shelf, stage the robot to it, verify in the base-only run |
| Grasp miss of 2-5 cm | picks | teleop holes (round 2), servo (round 3) |
| Arm mount / EE-offset conventions (kinder gripper model, no-gripper executor model, `IKSolver(ee_offset=0.12)` on the arm server) | Cartesian replay | replay joints only; never the `ee_pos` stream |
| Long arm segments exceed executor iteration budgets | `ExecutionFailure` mid-plan | raise `max_iter_*` in the replay yaml; test in fake mode |
| Robotiq close crushes a soft object or misses a narrow one | grasp hold | pick objects with rigid ±x faces; set `gripper_close_position` per object |
| Base drive path grazes a floor object | base segments | the sim plans with `check_base_collisions`; keep `base_clearance`-style margin by staging objects ≥0.12 m apart as the validator asks |
| `NO PLAN FOUND` on the lab scene | planning | raise `--k-max`; choose heights that make section assignment unambiguous |

## 5. Feedback for the deploy kit (alphatamp)

- Works as described; `uv sync` was needed first (torch was not in the venv).
- `ROBOT_EXECUTION.md` calls `gripper` a finger opening in metres, but open values are
  0.2-0.31, which reads as a joint angle. The 0.01 closed threshold is right either way.
- On a 7-object scene with three talls (0.16, 0.14, 0.13) the ranker chose eight
  consecutive skeletons sharing the same proven-infeasible prefix
  (`pick(goal3) → place_short(goal3)`, goal3 = 0.13 m), each costing ~36 s before the
  refiner proved F3 at step 1 again. A proven failure at step k rules out every pool
  skeleton with the same first k+1 operators; skipping those without refining would have
  saved several minutes. Worth adding before a live run, since planning time is
  wall-clock time in the lab.
- A `robot_start` (SE2) override in `scene.yaml`, or a per-object `yaw`, would let staging
  error be absorbed in sim instead of by the tracker. Optional.

## 6. Order of work: everything off-robot first

Almost all of the software can be finished and tested away from the lab. The replay agent
never needs hardware to be exercised: `mode=fake` runs the full Runner loop
(agent → segmentation → executors → `FakeInterface`) on a laptop, and a real Level-B npz
already exists (`outputs/demo6/plan_level_b.npz`, 290 KB — small enough to commit as a
test fixture). The one genuinely lab-bound input is the shelf measurement
(`shelf_yaw_map`, board gaps, whether the pegs adjust); until then the plan carries the
assumed values (shelf at map `(1.5, 1.5)`, front facing −y, gaps 0.254) as explicit
parameters that a five-minute measurement later confirms or replaces.

### 6.1 Off-robot (do now, laptop only)

Software, in dependency order — each step ends with a fake-mode or pytest check:

1. `NpzPlanAgent` + the agent `_target_` change in `pipeline.py` (section 2.5, items 1-2).
   Unit-test the npz→pairs conversion and the home→map SE2 transform (states transformed,
   deltas rotated) against hand-computed cases; test with `home_yaw_map ≠ 0` since that is
   the case the lab will hit.
2. `conf/env/restock3d-replay.yaml` (section 2.5, items 3-4). Run the demo6 npz end to end
   in `mode=fake`: this settles the iteration budgets, the gripper decoding, the
   arm-settle-to-`joints[0]` prelude, and the preview/floor-bounds path with zero risk.
   Make it a pytest alongside `tests/test_pipeline.py` so it stays green.
3. `scripts/export_restock_scene.py` with `--from-json`, tested against a canned
   marker-detector payload (grab one from any recent cylinder-shelf log, or synthesize
   the documented `{"poses", "targets"}` shape).
4. The laptop-side orchestrator (section 3, item 1), with the perceive and execute legs
   stubbed by `--from-json` and `mode=fake`.
5. Round-2 grasp-hole machinery (section 3, item 2): the npz→pairs converter grows a
   `--gap-at-grasps` option that cuts each pick at the pre-grasp state and emits a
   `SkillCall` with the post-stow `predicted_state`; `SettleGapExecutor` handles it in
   fake mode, so the whole path is testable without a human or a robot.
6. Check whether `kortex_wheels/` has (or Kinova ships) a Python 3.11 wheel — ten minutes
   that decides how hard section 3 item 5 will be later. **Checked 2026-08-31: the wheel
   is `py3-none-any` (pure Python), so it is not what caps `requires-python` at <3.11;
   lifting the cap is a dependency-resolution exercise, not a Kinova blocker.**

Physical preparation that does not need the robot:

7. Choose and measure the 6 objects at a desk (heights/widths per section 2.1 step 3;
   calipers or a ruler). Fix `gripper_close_position` per object by width now.
8. Print the target-marker sheets (ids 35-46 exist under `conf/calibration/`; use
   `create_calibration_markers.py` if more are needed).
9. Author the lab scene at the desk and plan it: instead of perceiving object positions on
   site, pick the six floor positions now, write `scenes/lab_run1/scene.yaml`, run
   `deploy.py --render`, and review `plan.mp4`. The lab is then staged to the scene rather
   than the scene fit to the lab: compute each object's map-frame floor position through
   the section-2.2 transform and print a one-page staging sheet (marker id → tape-measure
   coordinates from two floor-marker reference points, plus the robot's start pose).
   Planning minutes are spent at the desk, not in the lab, and perception on site becomes
   a verification step rather than an input. If the shelf measurement later changes
   `shelf_yaw_map` or the gaps, only the transform and the staging sheet change — the
   scene, the plan, and the npz survive as long as object heights respect the 0.15 m cap.
10. Dry-run the replay yaml against `scenes/lab_run1`'s npz in fake mode (same as step 2
    but with the real payload for the day).

### 6.2 In the lab (target: one visit, ~half a day)

1. Shelf: reinstall the slot-1 board, measure the board surfaces and gaps, record
   `shelf_yaw_map`. If anything differs from the assumed values, update the transform
   constants and reprint the staging sheet (nothing else changes).
2. Tape the markers at the staging-sheet coordinates; set the objects on them; stage the
   robot at the printed start pose.
3. Run `export_restock_scene.py` live once — not to build the scene, but to verify the
   staged positions agree with the planned ones (print the per-object residuals; a few
   centimetres is fine, re-tape anything worse).
4. Base-only run (arm faked), then the air run, then the full run with e-stop in reach
   (section 2.6). The software has all been fake-mode green before arrival, so debugging
   on site should be limited to physical effects: staging error, grasp misses, reach at
   full extension.
5. Record the ceiling video on the successful run; compose with `plan.mp4` afterwards.

If the full run's grasps miss, the round-2 grasp-hole build from step 5 is already tested:
re-run with `--gap-at-grasps` and `env.real_gap=teleop` in the same visit. Round 3 (front-
grasp visual servo) and the in-process SPECTRE agent remain follow-ups after the video.
