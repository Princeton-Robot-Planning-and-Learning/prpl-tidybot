# PRPL Tidybot

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-tidybot/actions/workflows/ci.yml/badge.svg)

Code for the real TidyBot++ robot and a real-to-sim-to-real pipeline that
lets agents written against a kinder simulator drive the real environment
(or, for development, a `FakeInterface`).

## Pipeline shape

```
                  ┌────────────────────────────┐
                  │      RealTidyBotEnv        │
                  │ (gymnasium.Env wrapping    │
                  │      an Interface)         │
                  └────────────┬───────────────┘
                               │ TidyBotObservation
                               ▼
                  ┌────────────────────────────┐
                  │    PrplLab3DPerceiver      │
                  └────────────┬───────────────┘
                               │ ObjectCentricState
                               ▼
                  ┌────────────────────────────┐
                  │       Agent                │
                  │ (anything implementing     │
                  │  prpl_utils.gym_agent)     │
                  └────────────┬───────────────┘
                               │ 11-d kinematic3d action
                               ▼
                  ┌────────────────────────────┐
                  │  PrplLab3DActionGrounder   │
                  └────────────┬───────────────┘
                               │ TidyBotAction
                               ▼
                       (back to RealTidyBotEnv)
```

`prpl_utils.real_sim.Runner` glues these together. The base `Interface`
abstracts the underlying world; `FakeInterface` is the development
default. Real hardware support lands later.

## Install

```bash
uv sync --all-extras --dev
```

## Try it

```bash
python scripts/demo_real_to_sim_to_real.py --steps 10 --seed 0
```

## Develop

```bash
./run_ci_checks.sh          # autoformat, mypy, pylint, pytest
black --check . && isort --check-only .   # mirrors the CI autoformat job
```
