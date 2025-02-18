#!/bin/bash

# --- Environment Variables and Paths ---
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/datasets"
export HF_HUB_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/hub"
export XDG_CACHE_HOME="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/bigcodebench/.cache"
# (Do not change HOME; it is needed for your conda env to work)


export OPENAI_API_KEY=$(cat /home/chengyud/.config/openai/openai.conf)


cd openai
python openai_run.py

