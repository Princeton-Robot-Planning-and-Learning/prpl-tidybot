#!/usr/bin/env bash
# Launch a command on a remote host with the repo venv activated.
# Usage: scripts/run_remote.sh <ssh-target> <command...>

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ssh-target> <command...>" >&2
    exit 64
fi

host="$1"
shift
repo_dir="${PRPL_REPO_DIR:-~/prpl-tidybot}"
branch="${PRPL_BRANCH:-}"

# `uv`'s installer drops the binary in ~/.local/bin/, which is typically only
# added to PATH by the user's interactive shell rc (~/.bashrc). The SSH command
# below runs a non-interactive non-login shell that doesn't source any rc file,
# so `uv` isn't visible without this explicit prepend. The escaped `\$HOME` is
# evaluated on the remote, not locally.
remote_path_prefix='export PATH="$HOME/.local/bin:$PATH" && '

# Bootstrap chained into the SSH command via `&&` so any step's failure
# aborts the pane instead of starting the server (or shell) against
# stale code or a stale venv.
#
# 1. Optional git sync. When PRPL_BRANCH is set, force the remote
#    checkout into exact alignment with origin/<branch>. The remote
#    machines are NOT development boxes — they're meant to be
#    ephemeral mirrors of whatever the laptop most recently pushed —
#    so origin is unambiguously the source of truth and the local
#    branch must always re-snap to it (also recovers from force-pushes,
#    issue #44).
#
#    Before snapping, detect and refuse uncommitted modifications or
#    untracked files (`git status --porcelain` non-empty), so local
#    edits are never silently discarded. Local commits are not checked:
#    doing so via `git rev-list HEAD --not origin/<branch>` produced
#    false positives when concurrent panels raced on the same repo and
#    the remote-tracking ref briefly resolved to a stale value.
#
#    `git checkout -B <branch> origin/<branch>` always (re)creates the
#    local branch pointing at origin/<branch>, regardless of any prior
#    state of the local branch — broken upstream config, stale value,
#    or no local branch at all. Replaces an older
#    `git checkout <branch> && git reset --hard origin/<branch>`
#    sequence that could leave the local branch's upstream tracking
#    config pointing at a ref that briefly didn't resolve (causing
#    "fatal: ambiguous argument 'origin/<branch>'") after a partial
#    failure on a previous run.
#
#    `git fetch origin <branch>` rather than `git pull --ff-only` so
#    we don't depend on `branch.<name>.merge` config that may be
#    missing on a fresh remote checkout and then errors out with
#    "Cannot fast-forward to multiple branches".
# 2. Always `uv sync`. Picks up any pyproject.toml / uv.lock changes
#    from the just-synced branch (or from a manual `git pull` on the
#    remote since the last launch) before any Python code runs. No-op
#    when nothing has changed.
#
# Concurrent-panel handling: set PRPL_SKIP_SYNC=1 on every panel
# except the one that should do the syncing. The sync panel holds an
# exclusive flock on .git/run_remote.lock for the duration of git +
# uv + opencv setup; the skipping panels acquire a shared flock on the
# same file (which blocks until the exclusive holder releases) and
# then go straight to launching the command. This way the skippers
# never run their command against a half-synced tree or venv.
if [[ -n "${PRPL_SKIP_SYNC:-}" ]]; then
    sync="echo 'run_remote.sh: PRPL_SKIP_SYNC set; waiting for the sync pane to finish before launching...' && (flock -s 9) 9>.git/run_remote.lock && echo 'run_remote.sh: sync pane finished; launching.' && "
elif [[ -n "$branch" ]]; then
    sync="(flock -x 9 && git fetch origin $branch && if [ -n \"\$(git status --porcelain)\" ]; then echo 'run_remote.sh: ERROR: remote checkout has uncommitted modifications or untracked files; refusing to overwrite. Resolve manually before re-launching.' >&2; git status --short >&2; exit 1; fi && git checkout -B $branch origin/$branch && uv sync && uv pip install --force-reinstall --no-deps opencv_wheels/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl) 9>.git/run_remote.lock && "
else
    sync="uv sync && uv pip install --force-reinstall --no-deps opencv_wheels/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl && "
fi

# -tt forces a PTY in both directions so closing the local pane delivers
# SIGHUP to the remote shell, killing the Python process cleanly.
if [[ "$*" == "bash" ]]; then
    # Interactive shell: use --rcfile so .bashrc loads first and the venv
    # activation runs after, otherwise .bashrc resets PS1 (and possibly
    # PATH) and the venv prefix disappears from the prompt.
    exec ssh -tt "$host" "${remote_path_prefix}cd $repo_dir && ${sync}exec bash --rcfile <(echo \"[ -f ~/.bashrc ] && source ~/.bashrc; cd $repo_dir; source .venv/bin/activate\")"
else
    exec ssh -tt "$host" "${remote_path_prefix}cd $repo_dir && ${sync}source .venv/bin/activate && exec $*"
fi
