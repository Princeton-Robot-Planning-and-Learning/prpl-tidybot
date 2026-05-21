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
#    so origin is unambiguously the source of truth and a hard reset
#    is the right tool to recover from force-pushes (issue #44).
#
#    But blindly hard-resetting would silently destroy any local work
#    if someone did happen to edit / commit on the remote. So first
#    detect any deviation and refuse loudly:
#      * `git status --porcelain` non-empty → uncommitted modifications
#        or untracked files.
#      * `git rev-list HEAD --not origin/<branch>` non-empty → local
#        commits the remote tip doesn't have.
#    If either trips, the bootstrap aborts with a clear message and
#    the operator has to resolve manually. If both pass, `git reset
#    --hard origin/<branch>` brings the checkout into exact alignment.
#
#    Using `origin/<branch>` (a remote-tracking ref that persists)
#    rather than `FETCH_HEAD` (a transient single-file ref that's
#    brittle across intervening git commands): an earlier draft used
#    FETCH_HEAD and surfaced "fatal: ambiguous argument 'FETCH_HEAD'"
#    failures on the NUC for reasons we didn't fully trace.
#    `origin/<branch>` is updated by the preceding `git fetch origin
#    <branch>` either way and is immune to that class of issue.
#
#    `git fetch origin <branch>` rather than `git pull --ff-only` so
#    we don't depend on `branch.<name>.merge` config that may be
#    missing on a fresh remote checkout and then errors out with
#    "Cannot fast-forward to multiple branches".
# 2. Always `uv sync`. Picks up any pyproject.toml / uv.lock changes
#    from the just-synced branch (or from a manual `git pull` on the
#    remote since the last launch) before any Python code runs. No-op
#    when nothing has changed.
if [[ -n "$branch" ]]; then
    sync="git pack-refs --prune && git fetch --prune origin $branch && if [ -n \"\$(git status --porcelain)\" ]; then echo 'run_remote.sh: ERROR: remote checkout has uncommitted modifications or untracked files; refusing to overwrite. Resolve manually before re-launching.' >&2; git status --short >&2; exit 1; fi && git checkout $branch && if [ -n \"\$(git rev-list HEAD --not origin/$branch)\" ]; then echo 'run_remote.sh: ERROR: remote HEAD has commits not on origin/$branch; refusing to discard them. Resolve manually before re-launching.' >&2; git log HEAD --not origin/$branch --oneline >&2; exit 1; fi && git reset --hard origin/$branch && uv sync --no-install-package opencv-python --no-install-package opencv-contrib-python && "
else
    sync="uv sync --no-install-package opencv-python --no-install-package opencv-contrib-python && "
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
