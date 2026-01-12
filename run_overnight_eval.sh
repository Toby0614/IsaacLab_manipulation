#!/bin/bash
# =============================================================================
# OVERNIGHT DROPOUT EVALUATION SCRIPT
# =============================================================================
# This script evaluates all M1-M4 policies under both:
# - Variant 1: Phase-based dropout (dropout at specific manipulation phases)
# - Variant 2: Time-based dropout (dropout at specific onset times)
#
# Results are saved to: results/dropout_eval/
# Estimated runtime: 4-8 hours depending on GPU
# =============================================================================

# Exit on errors, undefined vars; and ensure failures inside pipelines (e.g., python | tee) fail the script.
set -euo pipefail

cd /home/toby0614/IsaacLab/Projects/Manipulation_policy

# Create output directory
mkdir -p results/dropout_eval_tight

# Log file
LOG_FILE="results/dropout_eval/eval_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"

# Common evaluation parameters
# NOTE: Episodes complete in ~15 steps, so use tight onset/duration values
EVAL_ARGS="--eval_dropout \
    --eval_variant both \
    --eval_phases reach,grasp,lift,transport,place \
    --eval_onset_steps 1,3,5,7,9,11,13 \
    --eval_durations 1,2,3,5,7,10,15 \
    --eval_episodes 100 \
    --eval_output_dir results/dropout_eval_tight \
    --num_envs 64 \
    --headless \
    --enable_cameras"

echo "======================================" | tee -a "$LOG_FILE"
echo "Starting dropout robustness evaluation..." | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Full path to policy directories
POLICY_BASE="/home/toby0614/IsaacLab/Projects/Manipulation_policy/logs/rsl_rl/franka_pickplace"

# Manually pinned checkpoints (avoid any "latest checkpoint" auto-resolution)
M1_CKPT="${POLICY_BASE}/M1_policy/model_1000.pt"
M2_CKPT="${POLICY_BASE}/M2_policy/model_1500.pt"
M3_CKPT="${POLICY_BASE}/M3_policy/model_4000.pt"
M4_CKPT="${POLICY_BASE}/M4_policy/model_1000.pt"

# Sanity-check checkpoints exist before starting long eval.
for ckpt in "$M1_CKPT" "$M2_CKPT" "$M3_CKPT" "$M4_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERROR] Missing checkpoint: $ckpt" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# Evaluate M1 (Vision + Proprio)
echo "" | tee -a "$LOG_FILE"
echo ">>> Evaluating M1_policy..." | tee -a "$LOG_FILE"
python scripts/rsl_rl/play.py \
    --task Isaac-Franka-PickPlace-v0 \
    --checkpoint "${M1_CKPT}" \
    --policy_name M1_policy \
    $EVAL_ARGS \
    2>&1 | tee -a "$LOG_FILE"

# Evaluate M2 (Vision + Proprio + Force) - NOTE: needs --force_sensing
echo "" | tee -a "$LOG_FILE"
echo ">>> Evaluating M2_policy (with force sensing)..." | tee -a "$LOG_FILE"
python scripts/rsl_rl/play.py \
    --task Isaac-Franka-PickPlace-v0 \
    --checkpoint "${M2_CKPT}" \
    --policy_name M2_policy \
    --force_sensing \
    --force_mode grasp_indicator \
    $EVAL_ARGS \
    2>&1 | tee -a "$LOG_FILE"

# Evaluate M3 (Vision + Proprio, dropout-trained)
echo "" | tee -a "$LOG_FILE"
echo ">>> Evaluating M3_policy..." | tee -a "$LOG_FILE"
python scripts/rsl_rl/play.py \
    --task Isaac-Franka-PickPlace-v0 \
    --checkpoint "${M3_CKPT}" \
    --policy_name M3_policy \
    $EVAL_ARGS \
    2>&1 | tee -a "$LOG_FILE"

# Evaluate M4 (Vision + Proprio + Force, dropout-trained) - NOTE: needs --force_sensing
echo "" | tee -a "$LOG_FILE"
echo ">>> Evaluating M4_policy (with force sensing)..." | tee -a "$LOG_FILE"
python scripts/rsl_rl/play.py \
    --task Isaac-Franka-PickPlace-v0 \
    --checkpoint "${M4_CKPT}" \
    --policy_name M4_policy \
    --force_sensing \
    --force_mode grasp_indicator \
    $EVAL_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "Evaluation complete!" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Generate visualizations (no Isaac Sim needed)
echo "" | tee -a "$LOG_FILE"
echo "Generating visualizations..." | tee -a "$LOG_FILE"
python scripts/visualize_sensitivity_map.py \
    --results_dir results/dropout_eval_tight \
    --output_dir results/figures \
    --variant both \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "All done! Check:" | tee -a "$LOG_FILE"
echo "  - results/dropout_eval/ for raw data (JSON, CSV)" | tee -a "$LOG_FILE"
echo "  - results/figures/ for heatmaps and plots" | tee -a "$LOG_FILE"

