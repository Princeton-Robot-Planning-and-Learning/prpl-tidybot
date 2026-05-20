#!/usr/bin/env bash
# Tear down the TidyBot servers and the tmuxinator session.
# Verifies each step and exits non-zero if anything survived.

set -uo pipefail

if [ "${PRPL_LAB:-prpl}" = "fwing" ]; then
    _nuc_default="tidybot-nuc"
    _perc_default="tidybot-laptop"
else
    _nuc_default="tidybot-nuc-prpl"
    _perc_default="tidybot-laptop-prpl"
fi
NUC="${PRPL_NUC_HOST:-$_nuc_default}"
PERC="${PRPL_PERCEPTION_HOST:-$_perc_default}"
SESSION="${PRPL_TMUX_SESSION:-prpl-tidybot}"

failed=0

# Remote script: kills python processes matching $1, then prints any survivors
# (filtered by comm == python so the bash shell running this script — whose
# argv contains the pattern — doesn't show up as a survivor).
read -r -d '' REMOTE_SCRIPT <<'REMOTE' || true
pattern="$1"
is_python() {
    case "$(cat /proc/"$1"/comm 2>/dev/null)" in
        python*) return 0 ;;
    esac
    return 1
}
for pid in $(pgrep -f "$pattern"); do
    is_python "$pid" && kill "$pid" 2>/dev/null
done
sleep 1
for pid in $(pgrep -f "$pattern"); do
    if is_python "$pid"; then
        tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null
        echo
    fi
done
REMOTE

err() {
    echo "ERROR: $*" >&2
    failed=1
}

# Kill processes matching <pattern> on <target>, then confirm none remain.
# Catches SSH connect failures as well as stuck processes.
#
# The script body goes over stdin via `bash -s` (so newlines in the script
# don't get word-split by ssh's arg-joining). We then filter survivors by
# comm == python — the bash shell doing the kill matches the pattern in its
# own argv, but its comm is `bash`, so it's excluded from both the kill
# pass and the survivor report.
kill_remote() {
    local target="$1" pattern="$2" out rc
    echo "Stopping '$pattern' on $target..."
    out=$(ssh -o ConnectTimeout=10 "$target" bash -s "$pattern" <<<"$REMOTE_SCRIPT" 2>&1)
    rc=$?
    if [[ $rc -ne 0 ]]; then
        err "ssh to $target failed (exit $rc):"
        if [[ -n "$out" ]]; then
            printf '%s\n' "$out" | sed 's/^/  /' >&2
        else
            echo "  (no output — likely auth failure; run 'ssh $target true' to debug)" >&2
        fi
        return
    fi
    if [[ -n "$out" ]]; then
        err "processes matching '$pattern' still running on $target:"
        printf '%s\n' "$out" | sed 's/^/  /' >&2
    fi
}

kill_remote "$NUC" "third_party.base_server"
kill_remote "$PERC" "marker_detector"

echo "Killing tmux session '$SESSION'..."
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        err "tmux session '$SESSION' still exists after kill-session"
    fi
else
    echo "  (no session named '$SESSION')"
fi

if [[ $failed -ne 0 ]]; then
    echo >&2
    echo "FAILED: one or more cleanup steps did not complete." >&2
    exit 1
fi

echo "All clean."
