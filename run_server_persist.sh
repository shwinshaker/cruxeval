#!/bin/bash

#SBATCH -p interactive
#SBATCH -A llmservice_fm_text
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=2
#SBATCH --dependency=singleton
#SBATCH --job-name=llmservice_nlp_fm:eval-gpt-7b-1t-mistral-test-mbpp

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_BLOCKING_WAIT=0

#IMAGE_PATH="/lustre/fs1/portfolios/llmservice/users/dnarayanan/images/adlr+megatron-lm+pytorch+24.01-py3-draco_cw_ub_tot-te.sqsh"
IMAGE_PATH="/lustre/fsw/portfolios/llmservice/users/chengyud/images/adlr+megatron-lm+pytorch+24.01v3-py3-train-inf.sqsh"

EXPERIMENT="gpt-7b-1t-mistral"
OUTPUT_ROOT="/lustre/fsw/portfolios/llmservice/users/chengyud/experiments_evaluation/cruxeval"
# OUTPUT_ROOT="/lustre/fsw/portfolios/llmservice/users/chengyud/lm-evaluation-harness"
MODEL_CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/chengyud/runs/pretraining/gpt-7b-1t-mistral/checkpoints"
# LM_EVAL_HARNESS_PATH="/lustre/fsw/portfolios/llmservice/users/chengyud/lm-evaluation-harness"

########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
NAME="${EXPERIMENT}"
# if [ -n "${SLURM_JOB_ID:-}" ] ; then
#     SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
#     ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
# else
#     SCRIPT_PATH=$(realpath "$0")
#     ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
# fi

# SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
# REPO_DIR=$(python3 - <<'EOF' $SCRIPT_DIR
# # Searches for the repository directory by finding a directory containing
# # "megatron" in the parents of the first argument.

# import pathlib
# import sys

# path = pathlib.Path(sys.argv[1])
# while True:
#     if path.parent == path:
#         break
#     if (path / "megatron").exists():
#         print(path)
#     path = path.parent
# EOF
# )

SCRIPT_PATH=$(realpath "$0")
ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
REPO_DIR="/lustre/fsw/portfolios/llmservice/users/chengyud/eval-code"

RUN_DIR="${OUTPUT_ROOT}/runs/${NAME}"
# CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR}"
DATACACHE_DIR="${RUN_DIR}/data-cache"
LOGS_DIR="${RUN_DIR}/eval-logs"
EVAL_DIR="${RUN_DIR}/eval-results"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p "${LOGS_DIR}" "${EVAL_DIR}"

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE_PATH=${IMAGE_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}


TOKENIZER_MODEL=/lustre/fsw/portfolios/llmservice/users/chengyud/tokenizers/multiMixV5_fix_default_500000_128k.vocab.json
TIKTOKEN_PATTERN=v1

# Copy scripts.
mkdir -p ${RUN_DIR}/scripts
cp ${SCRIPT_PATH} ${RUN_DIR}/scripts

export NCCL_ALGO=^NVLS

BATCH_SIZE_PER_INSTANCE=32
MAX_TOKENS_TO_OOM=256000  # 64000 

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    # --deterministic-mode \
options=" \
    --max-tokens-to-oom ${MAX_TOKENS_TO_OOM} \
    --distributed-timeout-minutes 60 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --swiglu \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size ${BATCH_SIZE_PER_INSTANCE} \
    --global-batch-size 1536 \
    --train-samples 73242187 \
    --lr-decay-samples 73242187 \
    --valid-data-path 1.0 /lustre/fsw/portfolios/llmservice/users/bnorick/data/binidx/val/val \
    --test-data-path 1.0 /lustre/fsw/portfolios/llmservice/users/bnorick/data/binidx/val/val \
    --lr 4.5e-5 \
    --min-lr 4.5e-7 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type TikTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tiktoken-pattern ${TIKTOKEN_PATTERN} \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 4000 \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.0134 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --use-distributed-optimizer \
    --manual-gc \
    --num-workers 1 \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# --bf16 \

run_cmd="python -u ${REPO_DIR}/tools/run_text_generation_server.py ${options}"

# Start the server
echo "[$(date)] Starting the inference server"
srun --output=${LOGS_DIR}/%x_%j_${DATETIME}.log --export=ALL -l \
    --container-image ${IMAGE_PATH} \
    --container-mounts "/home:/home,/lustre:/lustre" \
    --no-container-mount-home \
    sh -c "pip install boto3 flask flask-restful tiktoken && cd ${DIR} && ${run_cmd}" &

sleep 4h


