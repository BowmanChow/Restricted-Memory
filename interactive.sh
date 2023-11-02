#!/bin/bash
# conda activate /sw/external/python/anaconda3_gpu

srun --account=bbsh-delta-gpu --partition=gpuA40x4-interactive \
  --nodes=1 --gpus-per-node=1 --tasks=1 \
  --tasks-per-node=1 --cpus-per-task=32 --mem=156g \
  --pty bash