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
  --checkpoint /home/toby0614/IsaacLab/Projects/Manipulation_policy/logs/rsl_rl/franka_pickplace/2026-01-08_22-58-04/model_4000.pt