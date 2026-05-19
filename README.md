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
                  │           Agent            │
                  │ (e.g. BilevelPlanningAgent)│
                  └────────────┬───────────────┘
                               │ 11-d kinder action
                               ▼
                  ┌────────────────────────────┐
                  │      ActionGrounder        │
                  └────────────┬───────────────┘
                               │ real_action
                               ▼
                       (back to real_env)
```

`prpl_utils.real_sim.Runner` glues these together. The base `Interface`
abstracts the underlying world; `FakeInterface` is the dev default and
`RealInterface` stubs hardware until it gets wired up.

## Install

```bash
uv sync --all-extras --dev
```

## Try it

```bash
python scripts/run_planner.py env=base_motion3d mode=sim
python scripts/run_planner.py env=base_motion3d mode=fake max_eval_steps=200
python scripts/run_planner.py env=prpl3d-o1 mode=sim seed=42
python scripts/run_planner.py env=base_motion3d mode=real     # raises NotImplementedError
                                                              # from the first
                                                              # RealInterface read
```

Each env yaml under `conf/env/` declares all three pipelines
(`fake` / `sim` / `real`); pick one with `mode=...`. **Adding a new env
is one new yaml file** — there is no env switch in code. Hydra composes
the perceiver, action grounder, and env wrapper from the yaml's
`_target_` references.

## Develop

```bash
./run_ci_checks.sh                              # autoformat, mypy, pylint, pytest
pytest tests/test_pipeline.py                   # smoke-test the planner pipeline
```
