#!/usr/bin/env bash
# Sequential training queue for the single-GPU box (one RTX 5090 -> jobs must
# run one at a time, never in parallel). Each experiment runs in the
# foreground (--no-tmux) so this script naturally blocks between jobs; the
# whole script should itself be launched inside ONE outer tmux session so the
# queue survives SSH disconnects over a weekend.
#
# A failed job is logged and the queue moves on to the next one — one bad
# config shouldn't cost the rest of the weekend's GPU time.
#
# Usage:
#   tmux new-session -d -s weekend_queue \
#     './docker/queue_train.sh exp1 exp2 exp3'
#   tmux attach -t weekend_queue   # to watch; Ctrl-b d to detach again
#
# Progress/results also stream to Weights & Biases as each job trains
# (project cyclo-lerobot) — no need to stay attached to check on it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENTS_DIR="$REPO_ROOT/cyclo_brain/train/experiments"
QUEUE_DIR="$REPO_ROOT/docker/workspace/runs/_queue"
mkdir -p "$QUEUE_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="$QUEUE_DIR/${STAMP}_queue.log"
SUMMARY="$QUEUE_DIR/${STAMP}_summary.txt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <experiment> [<experiment> ...]" >&2
  exit 1
fi

log "Queue started with ${#} experiment(s): $*"
queue_start=$(date +%s)

for exp in "$@"; do
  job_start=$(date +%s)
  log "START $exp"

  "$REPO_ROOT/docker/container.sh" train-lerobot "$exp" --no-tmux
  rc=$?

  job_end=$(date +%s)
  elapsed_min=$(( (job_end - job_start) / 60 ))

  if [ "$rc" -ne 0 ]; then
    log "END   $exp -> FAILED (exit=$rc, ${elapsed_min}m)"
    echo "$exp: FAILED (exit=$rc, ${elapsed_min}m)" >> "$SUMMARY"
    continue
  fi

  log "END   $exp -> OK (${elapsed_min}m)"

  run_name="$(grep -m1 '^name:' "$EXPERIMENTS_DIR/$exp.yaml" 2>/dev/null | sed -E 's/^name:[[:space:]]*//')"
  run_log="$REPO_ROOT/docker/workspace/runs/$run_name/train.log"
  {
    echo "=== $exp (run: $run_name, ${elapsed_min}m) ==="
    if [ -f "$run_log" ]; then
      python3 "$REPO_ROOT/docker/workspace/lerobot/analyze_act_loss.py" "$run_log" --interval 10000 2>&1
    else
      echo "(train.log not found at $run_log)"
    fi
    echo
  } >> "$SUMMARY"
done

queue_end=$(date +%s)
total_min=$(( (queue_end - queue_start) / 60 ))
log "Queue finished. Total: ${total_min}m. Summary: $SUMMARY"
