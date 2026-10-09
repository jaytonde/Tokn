#!/usr/bin/env bash
# Goal-mode loop for OpenCode: runs the tokn-optimizer agent in fresh sessions,
# forever by default. It stops only when the agent reports GOAL_BLOCKED, when
# MAX_ITERS is reached, or when you press Ctrl-C.
# OPTIMIZATION_LOG.md and git history are the agent's memory between sessions.
#
# Usage:  bash optimize_loop.sh               (runs until blocked or Ctrl-C)
#         MAX_ITERS=2 bash optimize_loop.sh   (stop after 2 sessions)
#
# --auto approves tool calls that are not explicitly denied, which is what makes
# this unattended. Review .opencode/agents/tokn-optimizer.md before running.
set -u
cd "$(dirname "$0")"
MAX_ITERS="${MAX_ITERS:-0}"   # 0 = no limit
LOG=OPTIMIZATION_LOG.md
RUNS=debug/opencode_runs
GIT_ID=(-c user.name="tokn-optimizer" -c user.email="tokn-optimizer@local")
mkdir -p "$RUNS"

# Refuse to start on uncommitted code: the loop resets leftovers between
# sessions and would throw away your work.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted changes in tracked files. Commit them first:"; git status --short --untracked-files=no
  exit 1
fi

# Kill benchmarks the agent may have left. The brackets stop pkill from matching this script.
cleanup_benchmarks() {
  pkill -f "[d]ebug/benchmark.py" 2>/dev/null
  pkill -f "[V]LLM::EngineCore" 2>/dev/null
  sleep 2
  pkill -f "[c]ompile_worker" 2>/dev/null
}
trap 'echo; echo "Stopping: killing leftover benchmarks"; pkill -f "[o]pencode run --agent tokn-optimizer"; cleanup_benchmarks; exit 130' INT TERM

save_log() {
  if [ -f "$LOG" ] && [ -n "$(git status --porcelain -- "$LOG")" ]; then
    git add "$LOG" && git "${GIT_ID[@]}" commit -q -m "$1"
  fi
}

# Continue session numbering across restarts.
last=$(ls "$RUNS"/iter_*.log 2>/dev/null | sed -E 's|.*/iter_0*([0-9]+)_.*|\1|' | sort -n | tail -1)
session=${last:-0}
ran=0

cleanup_benchmarks
while :; do
  if [ "$MAX_ITERS" -gt 0 ] && [ "$ran" -ge "$MAX_ITERS" ]; then
    echo "Reached MAX_ITERS=$MAX_ITERS"; exit 0
  fi
  ran=$((ran + 1)); session=$((session + 1))
  out="$RUNS/iter_$(printf '%03d' "$session")_$(date +%Y%m%d_%H%M%S).log"
  echo "=== session $session -> $out"
  opencode run --agent tokn-optimizer --auto --title "tokn-opt-$session" \
    "Session $session of the open-ended Tokn optimization goal. Read $LOG first, especially the 'Experiments tried' index, and git log. Do not repeat any experiment already listed. Follow your method: profile, one change, correctness gate, benchmark, keep or revert, log every experiment including failures, commit. Beating vLLM is a milestone, not the end. Do as many experiments as you can in this session. End with GOAL_PROGRESS or GOAL_BLOCKED." \
    2>&1 | tee "$out"

  cleanup_benchmarks
  save_log "log: session $session uncommitted entries (saved by optimize_loop)"

  if ! grep -q "GOAL_PROGRESS:\|GOAL_BLOCKED:" "$out"; then
    printf '\n## [loop] %s  Session %s ended without a status line\n- Outcome: SESSION CRASHED or cut off. Transcript: %s\n- Any experiment in progress was not finished. Check the transcript before retrying it.\n' \
      "$(date '+%Y-%m-%d %H:%M')" "$session" "$out" >> "$LOG"
    save_log "log: session $session ended without status"
  fi

  # Set aside leftover uncommitted code so the next session starts clean.
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    git "${GIT_ID[@]}" stash push -m "optimize_loop: leftover changes after session $session" >/dev/null
    echo "Stashed leftover code changes from session $session (see: git stash list)"
  fi

  if grep -q "GOAL_BLOCKED:" "$out"; then
    echo "Agent blocked in session $session:"; grep "GOAL_BLOCKED:" "$out" | tail -1; exit 1
  fi
done
