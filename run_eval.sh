#!/bin/bash

set -euo pipefail

cd /home/toby0614/IsaacLab/Projects/Manipulation_policy

OUT_DIR="results/pose_eval_final_fixed2"
mkdir -p "${OUT_DIR}"
POLICY_BASE="/home/toby0614/IsaacLab/Projects/Manipulation_policy/logs/rsl_rl/franka_pickplace"

M1_CKPT="${POLICY_BASE}/M1_new/model_2000.pt"
M2_CKPT="${POLICY_BASE}/M2_new/model_1000.pt"
M3_CKPT="${POLICY_BASE}/M3_pose/model_1500.pt"
M4_CKPT="${POLICY_BASE}/M4_pose/model_1500.pt"

for ckpt in "$M1_CKPT" "$M2_CKPT" "$M3_CKPT" "$M4_CKPT"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint: $ckpt"
    exit 1
  fi
done

POSE_EVAL_ARGS="--eval_pose_corruption \
  --pose_eval_variant both \
  --pose_eval_phases reach,grasp,lift,transport,place \
  --pose_eval_onset_steps 1,5,10,15,20,25,30,35 \
  --pose_eval_durations 5,10,20,40,80 \
  --pose_eval_modes hard,freeze,delay,noise \
  --pose_eval_episodes 500 \
  --pose_eval_require_triggered \
  --pose_eval_success_metric triggered \
  --pose_eval_max_attempts_multiplier 10 \
  --pose_eval_output_dir ${OUT_DIR} \
  --pose_eval_delay_steps 5 \
  --pose_eval_noise_std 0.01 \
  --pose_eval_drift_noise_std 0.001 \
  --num_envs 1600 \
  --headless \
  --enable_cameras"

echo "======================================================"
echo "POSE CORRUPTION EVAL (ALL POLICIES)"
echo "Output dir: ${OUT_DIR}"
echo "======================================================"

echo ""
echo ">>> Evaluating M1_policy ..."
python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --checkpoint "${M1_CKPT}" \
  --policy_name M1_policy \
  ${POSE_EVAL_ARGS}

echo ""
echo ">>> Evaluating M2_policy (force) ..."
python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --checkpoint "${M2_CKPT}" \
  --policy_name M2_policy \
  --force_sensing \
  --force_mode grasp_indicator \
  ${POSE_EVAL_ARGS}

echo ""
echo ">>> Evaluating M3_pose ..."
python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --checkpoint "${M3_CKPT}" \
  --policy_name M3_pose \
  ${POSE_EVAL_ARGS}

echo ""
echo ">>> Evaluating M4_pose (force) ..."
python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --checkpoint "${M4_CKPT}" \
  --policy_name M4_pose \
  --force_sensing \
  --force_mode grasp_indicator \
  ${POSE_EVAL_ARGS}

echo ""
echo "======================================================"
echo "DONE. Results saved to: ${OUT_DIR}"
echo "======================================================"


