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

# Bootstrap chained into the SSH command via `&&` so any step's failure
# aborts the pane instead of starting the server (or shell) against
# stale code or a stale venv.
#
# 1. Optional git sync. When PRPL_BRANCH is set, fast-forward the remote
#    checkout to that branch. Use `merge --ff-only FETCH_HEAD` rather
#    than `git pull --ff-only` — the latter depends on
#    `branch.<name>.merge` config that may be missing on a fresh remote
#    checkout and then errors out with "Cannot fast-forward to multiple
#    branches".
# 2. Always `uv sync`. Picks up any pyproject.toml / uv.lock changes
#    from the just-synced branch (or from a manual `git pull` on the
#    remote since the last launch) before any Python code runs. No-op
#    when nothing has changed.
if [[ -n "$branch" ]]; then
    sync="git fetch origin $branch && git checkout $branch && git merge --ff-only FETCH_HEAD && uv sync && "
else
    sync="uv sync && "
fi

# -tt forces a PTY in both directions so closing the local pane delivers
# SIGHUP to the remote shell, killing the Python process cleanly.
if [[ "$*" == "bash" ]]; then
    # Interactive shell: use --rcfile so .bashrc loads first and the venv
    # activation runs after, otherwise .bashrc resets PS1 (and possibly
    # PATH) and the venv prefix disappears from the prompt.
    exec ssh -tt "$host" "cd $repo_dir && ${sync}exec bash --rcfile <(echo \"[ -f ~/.bashrc ] && source ~/.bashrc; cd $repo_dir; source .venv/bin/activate\")"
else
    exec ssh -tt "$host" "cd $repo_dir && ${sync}source .venv/bin/activate && exec $*"
fi
