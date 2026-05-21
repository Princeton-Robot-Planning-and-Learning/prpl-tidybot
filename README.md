# PRPL Tidybot

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-tidybot/actions/workflows/ci.yml/badge.svg)

Code for the real TidyBot++ robot and a real-to-sim-to-real pipeline that
lets agents written against a kinder simulator drive the real environment
(or, for development, a `FakeInterface` or sim shadow).

## Pipeline shape

```
                  ┌────────────────────────────┐
                  │   real_env (gymnasium.Env) │
                  │ wrapping an Interface or   │
                  │ a kinder env directly      │
                  └────────────┬───────────────┘
                               │ obs
                               ▼
                  ┌────────────────────────────┐
                  │         Perceiver          │
                  └────────────┬───────────────┘
                               │ ObjectCentricState
                               ▼
                  ┌────────────────────────────┐
                  │       PlanningAgent        │
                  │ (e.g. BilevelPlanningAgent)│
                  └────────────┬───────────────┘
                               │ planned trajectory
                               │ [(state, action), …]
                               ▼
                  ┌────────────────────────────┐
                  │       PlanExecutor         │
                  │ (per-tick, closed loop)    │
                  └────────────┬───────────────┘
                               │ real_action
                               ▼
                       (back to real_env)
```

`prpl_utils.real_sim.Runner` glues these together: each outer `step()`
calls `agent.plan()` for a state-action trajectory, then ticks the
`PlanExecutor` against the real env until it reports done.
`FakeInterface` is the dev default; `RealInterface` is the real-robot
backend, with `arm_interface` / `camera_interface` overridable so a
base-only rollout can swap in fakes for the parts that aren't wired
up yet.

## Install

```bash
uv sync --all-extras --dev
```

## Try it

```bash
python scripts/run_planner.py env=base_motion3d mode=sim
python scripts/run_planner.py env=base_motion3d mode=fake max_eval_steps=200
python scripts/run_planner.py env=prpl3d-o1 mode=sim seed=42

# Drives the real base toward an ArUco target. Needs the tmuxinator stack
# running (see "Launching the servers" below); arm + cameras are faked for
# this env so only the base controller + marker detector are exercised.
python scripts/run_planner.py env=base_motion3d mode=real

# Uses the full RealInterface (real arm + cameras), which still raises
# NotImplementedError from RealArmInterface / RealCameraInterface — the
# arm wiring lands in #48.
python scripts/run_planner.py env=prpl3d-o1 mode=real
```

Each env yaml under `conf/env/` declares all three pipelines
(`fake` / `sim` / `real`); pick one with `mode=...`. **Adding a new env
is one new yaml file** — there is no env switch in code. Hydra composes
the perceiver, plan executor, and env wrapper from the yaml's
`_target_` references.

## Launching the servers

The real pipeline runs across three machines: a **robot NUC** (base RPC
server, port 50000), a **perception PC** (marker-detector server, port
6002), and an **orchestrator** (the box you sit at — could be a laptop or
the NUC itself). `.tmuxinator.yml` opens both backends in one tmux session
from the orchestrator.

### One-time setup on the orchestrator

```bash
gem install tmuxinator                          # tmux itself must already be installed

# Generate a key if you don't have one yet, then trust it on each remote.
ssh-keygen -t ed25519
ssh-copy-id tidybot@tidybot-nuc
ssh-copy-id yixuan@tidybot-laptop
```

The defaults assume `~/.ssh/config` aliases named `tidybot-nuc` and
`tidybot-laptop`. If yours are different, override the two env vars below.
Without passwordless SSH every tmuxinator pane prompts for a password and
the launcher is no faster than starting the servers by hand. Optionally
add `ControlMaster auto` / `ControlPath ~/.ssh/cm-%r@%h:%p` /
`ControlPersist 10m` to `~/.ssh/config` so the panes share one TCP
connection per remote.

### Configuration (env vars with defaults)

| Variable               | Default                  | What it points at                              |
| ---------------------- | ------------------------ | ---------------------------------------------- |
| `PRPL_NUC_HOST`        | `tidybot@tidybot-nuc`    | SSH target for the robot NUC.                  |
| `PRPL_PERCEPTION_HOST` | `yixuan@tidybot-laptop`  | SSH target for the perception PC.              |
| `PRPL_REPO_DIR`        | `~/prpl-tidybot`         | Repo checkout path on the *remote* machines.   |

### Launch

```bash
cd prpl-tidybot
./scripts/launch.sh                          # use whatever's already checked out remotely
./scripts/launch.sh my-feature-branch        # sync every remote pane to <branch> first
```

When a branch is given, each pane brings the remote checkout into exact
alignment with `origin/<branch>` before starting its server. The
**remote machines are not development boxes** — they're meant to be
ephemeral mirrors of whatever the laptop most recently pushed — so
`origin` is the source of truth. Concretely, each pane runs (roughly):

```bash
git fetch origin <branch>
# Refuse loudly if the remote checkout has been edited locally:
test -z "$(git status --porcelain)"        # no uncommitted modifications, no untracked files
test -z "$(git rev-list HEAD --not FETCH_HEAD)"   # no local commits ahead of origin
# If both pass: hard-reset onto the just-fetched tip.
git checkout <branch> && git reset --hard FETCH_HEAD
```

If either check fails, the pane prints what it found and aborts —
nothing on the remote gets overwritten silently. The fix is to SSH
in, deal with the work (`git stash`, push the commits somewhere,
whatever), and re-launch. If you find yourself doing this often, you
probably want to do that work on a real dev machine instead.

After the git sync, every pane also runs `uv sync` so that
`pyproject.toml` / `uv.lock` changes from the just-synced branch
land in the venv before any Python code starts.

The wrapper is a thin shim around `tmuxinator start ./.tmuxinator.yml`,
which (unlike `tmuxinator local`) forwards positional args to the
project ERB.

One window opens with three tiled panes:

```
┌──────────────┬──────────────┐
│ base_server  │ marker_      │
│  (NUC)       │  detector    │
│              │   (perception)│
├──────────────┴──────────────┤
│        planner shell        │
│  (NUC, .venv pre-activated) │
└─────────────────────────────┘
```

Move between panes with `Ctrl-b <arrow>` (or `Ctrl-b o` to cycle).
`Ctrl-b z` zooms the current pane to full-screen and back, handy when
you want a wider view of one log. Detach with `Ctrl-b d`.

To tear everything down — both remote server processes and the tmux
session — in one shot:

```bash
./scripts/stop_servers.sh
```

This is the recommended shutdown path: `tmux kill-session` alone relies
on SIGHUP-via-PTY propagation that occasionally leaves orphans (especially
multiprocessing children like the marker detector's `camera_server`
forks). The script explicitly `pkill`s each remote server first, then
closes the tmux session, and **verifies each step** — it exits non-zero
and prints which host/pattern/session survived if anything did, so you
can't accidentally leave the lab in a broken state.

In the `planner` window, run e.g.

```bash
python scripts/run_planner.py env=base_motion3d mode=real
```

## Develop

```bash
./run_ci_checks.sh                              # autoformat, mypy, pylint, pytest
pytest tests/test_pipeline.py                   # smoke-test the planner pipeline
```
