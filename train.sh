#!/bin/bash
# Franka Pick-and-Place Training


# M3_pose train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --pose_corruption \
  --pose_corruption_mode mixed


# M4 pose train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --force_sensing \
  --force_mode grasp_indicator \
  --pose_corruption \
  --pose_corruption_mode mixed

  # checkpoint
  tensorboard --logdir logs/rsl_rl/franka_pickplace/M2_new

# Evaluation
bash run_eval.sh


# success rate
M1 = 0.8717
M2 = 0.8933
M3 = 0.9446
M4 = 0.961