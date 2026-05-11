#!/usr/bin/env bash
# run_history_ablation.sh — train three models with different history window sizes.
#
# Each run trains from scratch for G_STEPS (30M) steps.
# Checkpoints → <exp_name>-td3_checkpoint.pth
# Logs        → history_ablation_logs/<exp_name>.log
#
# After all three finish, evaluate each with:
#   bash run_history_eval.sh
#
# Usage: bash run_history_ablation.sh

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
    if uv run python main.py "$@" 2>&1 | tee "$LOG_DIR/${name}.log"; then
        echo "  [$(date '+%H:%M:%S')]  DONE  : $name"
        PASSED+=("$name")
    else
        echo "  [$(date '+%H:%M:%S')]  FAILED: $name  (see $LOG_DIR/${name}.log)"
        FAILED+=("$name")
    fi
}

# k=1: no history — single frame only, cannot compare across timesteps
run fault_k1  --exp-name fault_k1  --obs-stack 1

# k=3: short history — ~60ms window at 50Hz control
run fault_k3  --exp-name fault_k3  --obs-stack 3

# k=10: long history — ~200ms window, diminishing returns expected
run fault_k10 --exp-name fault_k10 --obs-stack 10

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${#PASSED[@]}/3 training runs succeeded."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo ""
echo "  Checkpoints : fault_k{1,3,10}-td3_checkpoint.pth"
echo "  Logs        : $LOG_DIR/*.log"
echo ""
echo "  Next — evaluate each checkpoint:"
echo "    bash run_history_eval.sh"
echo "════════════════════════════════════════════════════════"
