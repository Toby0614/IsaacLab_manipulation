#!/bin/bash
# Franka Pick-and-Place Training

python scripts/rsl_rl/train.py \
    --task Isaac-Franka-PickPlace-v0 \
    --num_envs 1000 \
    --headless \
    --enable_cameras

cd /home/toby0614/IsaacLab/Projects/Manipulation_policy

python scripts/rsl_rl/play.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint /home/toby0614/IsaacLab/Projects/Manipulation_policy/logs/rsl_rl/franka_pickplace/2026-01-09_14-23-07/model_1000.pt

# M3 train
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
  --checkpoint logs/rsl_rl/franka_pickplace/M3_mix/model_4000.pt \
  --dropout_mode mixed

  #M2 train
python scripts/rsl_rl/train.py \
  --task Isaac-Franka-PickPlace-v0 \
  --num_envs 1600 \
  --headless \
  --enable_cameras \
  --force_sensing \
  --force_mode grasp_indicator

  