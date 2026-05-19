#!/usr/bin/env bash
# Launch the prpl-tidybot tmuxinator session, optionally syncing every
# remote pane to a specific branch first.
#
# Usage:
#   scripts/launch.sh                  # no branch sync
#   scripts/launch.sh <branch-name>    # `git fetch && checkout && pull --ff-only` per pane
#
# This is a thin wrapper around `tmuxinator start ./.tmuxinator.yml`,
# which (unlike `tmuxinator local`) forwards positional args to the
# project's ERB as `@args`.

set -euo pipefail

cd "$(dirname "$0")/.."

exec tmuxinator start ./.tmuxinator.yml ${1:+"$1"}
