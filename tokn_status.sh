#!/usr/bin/env bash
# Show what the tokn-optimizer agent has tried and what it got.
#
#   bash tokn_status.sh            one snapshot
#   bash tokn_status.sh -w         refresh every 30 s (Ctrl-C to quit)
#   bash tokn_status.sh -w 10      refresh every 10 s
#   bash tokn_status.sh -d 7       print full details of experiment [7]
#   bash tokn_status.sh -t         follow the live session transcript
set -u
cd "$(dirname "$0")"
LOG=OPTIMIZATION_LOG.md
RUNS=debug/opencode_runs

B=$'\e[1m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; C=$'\e[36m'; N=$'\e[0m'

# Print the section whose heading contains $1, up to the next heading of the same or higher level.
section() {
  awk -v name="$1" '
    /^#+ / {
      lvl = match($0, /[^#]/) - 1
      if (inside && lvl <= level) exit
      if (!inside && index(tolower($0), tolower(name))) { inside = 1; level = lvl; next }
    }
    inside { print }
  ' "$LOG"
}

colorize() {
  sed -E \
    -e "s/(KEPT|PASS|MILESTONE)/${G}\1${N}/g" \
    -e "s/(FAILED|FAIL|CRASHED|REVERTED)/${R}\1${N}/g" \
    -e "s/(ABANDONED|NOT RUN|NOT REACHED)/${Y}\1${N}/g"
}

latest_run() { ls -t "$RUNS"/iter_*.log 2>/dev/null | head -1; }

snapshot() {
  echo "${B}=== Tokn optimizer status  $(date '+%Y-%m-%d %H:%M:%S') ===${N}"
  if pgrep -f "[o]ptimize_loop.sh" >/dev/null; then echo "Loop: ${G}RUNNING${N}"; else echo "Loop: ${Y}not running${N}"; fi
  if pgrep -f "[d]ebug/benchmark.py" >/dev/null; then
    echo "Activity: ${C}benchmark in progress${N} ($(pgrep -af '[d]ebug/benchmark.py' | grep -o -- '--framework [a-z]*' | head -1))"
  fi
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{print "GPU: " $1 " busy, " $2 " / " $3}'

  if [ ! -f "$LOG" ]; then
    echo; echo "No $LOG yet. The agent creates it in its first session."
  else
    echo; echo "${B}--- Scoreboard (kept changes) ---${N}"
    section "Scoreboard" | sed '/^\s*$/d' | colorize
    echo; echo "${B}--- Experiments tried ---${N}"
    section "Experiments tried" | sed '/^\s*$/d' | colorize
    total=$(grep -cE '^## \[[0-9]+\]' "$LOG")
    kept=$(grep -E '^- Outcome:' "$LOG" | grep -c 'KEPT')
    failed=$(grep -E '^- Outcome:' "$LOG" | grep -cE 'FAILED|REVERTED|ABANDONED')
    crashes=$(grep -c '^## \[loop\]' "$LOG")
    echo; echo "${B}Totals:${N} $total experiments, ${G}$kept kept${N}, ${R}$failed failed or reverted${N}, $crashes crashed sessions"
  fi

  echo; echo "${B}--- Recent commits ---${N}"
  git log --oneline -12 --grep='^perf:' --grep='^log:' 2>/dev/null | colorize

  run=$(latest_run)
  if [ -n "$run" ]; then
    echo; echo "${B}--- Latest session: $run ---${N}"
    grep -E 'GOAL_PROGRESS:|GOAL_BLOCKED:' "$run" | tail -1
    echo "Last lines:"
    tail -n 8 "$run" | cut -c1-200
  fi
}

case "${1:-}" in
  -w) every="${2:-30}"; while :; do clear; snapshot; sleep "$every"; done ;;
  -d)
    [ -z "${2:-}" ] && { echo "usage: $0 -d <experiment number>"; exit 1; }
    awk -v n="$2" '
      index($0, "## [" n "]") == 1 { inside = 1; print; next }
      inside && /^##? / { exit }
      inside { print }
    ' "$LOG" | colorize ;;
  -t)
    run=$(latest_run)
    [ -z "$run" ] && { echo "No session transcripts yet."; exit 1; }
    echo "Following $run (Ctrl-C to stop)"; tail -f "$run" ;;
  *) snapshot ;;
esac
