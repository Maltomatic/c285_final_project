#!/usr/bin/env bash
# run_history_eval.sh — evaluate the three history-window checkpoints produced by
#                       run_history_ablation.sh (fault_k1, fault_k3, fault_k10).
#
# Each run loads its matching checkpoint and writes:
#   eval_logs/<exp_name>-eval_log.csv
#
# After all three finish, plot with:
#   uv run python eval_logs/history_grapher.py
#
# Usage: bash run_history_eval.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="history_ablation_logs"
mkdir -p "$LOG_DIR"

PASSED=()
FAILED=()

run() {
    local name="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')]  START : $name"
    echo "  uv run python main.py $*"
    echo "════════════════════════════════════════════════════════"
    if uv run python main.py "$@" 2>&1 | tee "$LOG_DIR/${name}_eval.log"; then
        echo "  [$(date '+%H:%M:%S')]  DONE  : $name"
        PASSED+=("$name")
    else
        echo "  [$(date '+%H:%M:%S')]  FAILED: $name  (see $LOG_DIR/${name}_eval.log)"
        FAILED+=("$name")
    fi
}

# k=1: no history
run fault_k1  --eval --exp-name fault_k1  --ckpt-name fault_k1  --obs-stack 1

# k=3: short history
run fault_k3  --eval --exp-name fault_k3  --ckpt-name fault_k3  --obs-stack 3

# k=10: long history
run fault_k10 --eval --exp-name fault_k10 --ckpt-name fault_k10 --obs-stack 10

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${#PASSED[@]}/3 eval runs succeeded."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo ""
echo "  Results : eval_logs/fault_k{1,3,10}-eval_log.csv"
echo "  Logs    : $LOG_DIR/*_eval.log"
echo ""
echo "  Plot results:"
echo "    uv run python eval_logs/history_grapher.py"
echo "════════════════════════════════════════════════════════"
