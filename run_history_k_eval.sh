#!/usr/bin/env bash
# Evaluate all 5 history-window checkpoints from modal_volume_test_1/.
# Default: 1-wheel fault, random placement, no jitter, alphas [0.0..0.5], injected at steps 150 & 700.
# Results → eval_logs/fault_k{1,3,5,7,10}-eval_log.csv

set -e

CKPT_DIR="modal_volume_test_1"
LOG_DIR="history_k_eval_logs"
mkdir -p "$LOG_DIR"

for K in 1 3 5 7 10; do
    echo "=== Evaluating fault_k${K} (obs-stack ${K}) ==="
    uv run python main.py \
        --eval \
        --exp-name   "fault_k${K}" \
        --ckpt-name  "${CKPT_DIR}/fault_k${K}" \
        --obs-stack  "${K}" \
        2>&1 | tee "${LOG_DIR}/fault_k${K}_eval.log"
    echo ""
done

echo "All evals done. CSVs in eval_logs/, logs in ${LOG_DIR}/."
echo ""
echo "Plot with:"
echo "  uv run python eval_logs/multi_exp_eval_grapher.py \\"
echo "      --experiment-name history_k \\"
echo "      --csv \"k1=eval_logs/fault_k1-eval_log.csv\" \\"
echo "      --csv \"k3=eval_logs/fault_k3-eval_log.csv\" \\"
echo "      --csv \"k5=eval_logs/fault_k5-eval_log.csv\" \\"
echo "      --csv \"k7=eval_logs/fault_k7-eval_log.csv\" \\"
echo "      --csv \"k10=eval_logs/fault_k10-eval_log.csv\""
