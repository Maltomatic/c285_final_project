#!/usr/bin/env bash
# run_ablation.sh — run all 9 ablation evaluation experiments sequentially.
#
# Results  → eval_logs/<exp_name>-eval_log.csv
# Stdout   → ablation_logs/<exp_name>.log
#
# NOTE: fault_w1 and fault_constant are identical experiments (1 wheel, no
#       jitter). They are run separately so each ablation section has its own
#       named CSV when referenced in ablation_grapher.py.
#
# Usage: bash run_ablation.sh

set -uo pipefail

# Always run from the project root regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="ablation_logs"
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
        echo "  [$(date '+%H:%M:%S')]  FAILED: $name  (see ablation_logs/${name}.log)"
        FAILED+=("$name")
    fi
}

# ── 1-3: Multi-wheel fault ablation (residual model) ─────────────────────────
run fault_w1          --eval --exp-name fault_w1          --ckpt-name fault
run fault_w2          --eval --exp-name fault_w2          --ckpt-name fault --num-fault-wheels 2
run fault_w3          --eval --exp-name fault_w3          --ckpt-name fault --num-fault-wheels 3

# ── 4-5: Same-side vs random placement ───────────────────────────────────────
run fault_w2_sameside --eval --exp-name fault_w2_sameside --ckpt-name fault --num-fault-wheels 2 --same-side
run fault_w3_sameside --eval --exp-name fault_w3_sameside --ckpt-name fault --num-fault-wheels 3 --same-side

# ── 6-7: Jitter ablation ─────────────────────────────────────────────────────
run fault_constant    --eval --exp-name fault_constant    --ckpt-name fault
run fault_jitter      --eval --exp-name fault_jitter      --ckpt-name fault --jitter-fault

# ── 8-9: Residual vs Pure RL under multi-wheel fault ─────────────────────────
run pure_w2           --eval --exp-name pure_w2           --ckpt-name fault_pure --pure --num-fault-wheels 2
run pure_w3           --eval --exp-name pure_w3           --ckpt-name fault_pure --pure --num-fault-wheels 3

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${#PASSED[@]}/9 experiments succeeded."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo ""
echo "  Results : eval_logs/*-eval_log.csv"
echo "  Logs    : ablation_logs/*.log"
echo "  Plot    : uv run python eval_logs/ablation_grapher.py"
echo ""
echo "  For per-alpha / per-wheel breakdowns on any subset, use:"
echo "    cd eval_logs && uv run python eval_grapher.py \\"
echo "      --experiment fault_w1=fault_w1-eval_log.csv \\"
echo "      --experiment fault_w2=fault_w2-eval_log.csv \\"
echo "      --experiment fault_w3=fault_w3-eval_log.csv"
echo "════════════════════════════════════════════════════════"
