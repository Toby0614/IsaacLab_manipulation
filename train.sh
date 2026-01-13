#!/bin/bash
# Franka Pick-and-Place Training

python scripts/rsl_rl/train.py \
    --task Isaac-Franka-PickPlace-v0 \
    --num_envs 1600 \
    --headless \
    --enable_cameras

cd /home/toby0614/IsaacLab/Projects/Manipulation_policy

python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint /home/toby0614/IsaacLab/Projects/Manipulation_policy/logs/rsl_rl/franka_pickplace/2026-01-09_14-23-07/model_1000.pt

# M3_vision train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --dropout_mode mixed

# M3 play
  python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint logs/rsl_rl/franka_pickplace/M3_pose/model_3500.pt

  #M2 train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --force_sensing \
  --force_mode grasp_indicator

  # M4 train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --dropout_mode mixed \
  --force_sensing \
  --force_mode grasp_indicator

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
bash run_pose_eval_all.sh