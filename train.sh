#!/bin/bash
# Franka Pick-and-Place Training

python scripts/rsl_rl/train.py \
    --task Isaac-Franka-PickPlace-v0 \
    --num_envs 1000 \
    --headless \
    --enable_cameras
