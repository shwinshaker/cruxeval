#! /bin/bash

set -eux

partition="cpu_short"
salloc \
  --job-name=llmservice_fm_text:dev \
  --account=llmservice_fm_text \
  --partition=${partition} \
  --nodes=1 \
  --time=04:00:00 \
  --exclusive

# --gpus-per-node=8 \
