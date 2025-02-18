#!/bin/bash
#SBATCH --job-name=bigcodebench_eval_cpu
#SBATCH --account=llmservice_fm_text
#SBATCH --partition=cpu_short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

source /lustre/fsw/portfolios/llmservice/users/chengyud/miniconda3/bin/activate cruxeval

# --- Environment Variables and Paths ---
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/datasets"
export HF_HUB_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/hub"
export XDG_CACHE_HOME="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/bigcodebench/.cache"
# (Do not change HOME; it is needed for your conda env to work)


# GENERATION_PATH="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/cruxeval/model_generations/gpt-3.5-turbo-0125_temp0.2_input/generations.json"
# GENERATION_SCORED_PATH="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/cruxeval/model_generations/gpt-3.5-turbo-0125_temp0.2_input/generations_scored.json"

GENERATION_DIR="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/cruxeval/runs/gpt-7b-1t-mistral/eval-results/bs_6/2025-02-17_16-33-21/megatron_temp0.2_input"
GENERATION_PATH="${GENERATION_DIR}/generations.json"
GENERATION_SCORED_PATH="${GENERATION_DIR}/generations_scored.json"

cd evaluation
python evaluate_generations.py \
    --generations_path ${GENERATION_PATH} \
    --scored_results_path ${GENERATION_SCORED_PATH} \
    --mode input
