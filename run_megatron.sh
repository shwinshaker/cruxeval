#!/bin/bash
#SBATCH --job-name=bigcodebench_eval_cpu
#SBATCH --account=llmservice_fm_text
#SBATCH --partition=cpu_short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

OUTPUT_ROOT="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/cruxeval"
MODEL_NAME="gpt-7b-1t-mistral"

RUN_DIR="${OUTPUT_ROOT}/runs/${MODEL_NAME}"
if [ ! -z ${CHECKPOINT_STEP} ]; then
    PADDED_CHECKPOINT_STEP=$(printf "%07d" ${CHECKPOINT_STEP})
    RUN_DIR="${RUN_DIR}/iter_${PADDED_CHECKPOINT_STEP}"
fi

LOGS_DIR="${RUN_DIR}/eval-logs"
EVAL_DIR="${RUN_DIR}/eval-results"
mkdir -p ${RUN_DIR}
mkdir -p ${LOGS_DIR}
mkdir -p ${EVAL_DIR}

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_FILE="${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${DATETIME}.log"

# --------------------------- Run the generation ---------------------------
echo
MEGATRON_IP=10.49.193.6  # 127.0.0.1
OUTPUT_PATH=${EVAL_DIR} 
source /lustre/fsw/portfolios/llmservice/users/chengyud/miniconda3/bin/activate cruxeval
# export HF_ALLOW_CODE_EVAL=1

# echo "[$(date)] Running all code benchmarks"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/datasets"
export HF_HUB_CACHE="/lustre/fsw/portfolios/llmservice/users/chengyud/huggingface/hub"
export XDG_CACHE_HOME="/lustre/fs1/portfolios/llmservice/users/chengyud/experiments_evaluation/bigcodebench/.cache"  # for `from appdirs import user_cache_dir` in `data.utils`
# export HOME="/lustre/fsw/portfolios/llmservice/users/chengyud"
# cannot set export HOME="", otherwise conda env doesn't work, not sure why

BENCH_TYPES=("input")  #  "output")
MODEL_PREFIX="megatron"
BATCH_SIZE=6  # 8 causes oom  # The server-side batch size will be batch_size * n_samples
DO_COT=false
if [[ "${DO_COT}" == "true" ]]; then
  MAX_NEW_TOKENS=1280
else
  MAX_NEW_TOKENS=64  # 128  # Need some tests, 10 is too small
fi
GREEDY=false  # true
N_SAMPLES=10  # number of trials in generation, should be 1 if greedy decoding
TEMPERATURE=0.2  # 1.0  # not used if greedy
base_url="http://${MEGATRON_IP}:5000/generate_until"

echo "[$(date)] << START GENERATION SETUP >>" |& tee -a ${LOG_FILE}
echo "BENCH_TYPES=${BENCH_TYPES[@]}" |& tee -a ${LOG_FILE}
echo "MODEL_PREFIX=${MODEL_PREFIX}" |& tee -a ${LOG_FILE}
echo "BATCH_SIZE=${BATCH_SIZE}" |& tee -a ${LOG_FILE}
echo "DO_COT=${DO_COT}" |& tee -a ${LOG_FILE}
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" |& tee -a ${LOG_FILE}
echo "GREEDY=${GREEDY}" |& tee -a ${LOG_FILE}
echo "N_SAMPLES=${N_SAMPLES}" |& tee -a ${LOG_FILE}
echo "TEMPERATURE=${TEMPERATURE}" |& tee -a ${LOG_FILE}
echo "<< END GENERATION SETUP >>" |& tee -a ${LOG_FILE}
echo -e "\n\n" |& tee -a ${LOG_FILE}

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_PATH="${OUTPUT_PATH}/bs_${BATCH_SIZE}/${DATETIME}"
echo "[$(date)] Starting generation.." |& tee -a ${LOG_FILE}
declare -A target_paths
cd megatron
for BENCH_TYPE in "${BENCH_TYPES[@]}"; do
  echo "---------------------------------------------" |& tee -a ${LOG_FILE}
  echo "[$(date)] bench_type: ${BENCH_TYPE}" |& tee -a ${LOG_FILE}
  python -u megatron_run.py \
        --root ${OUTPUT_PATH} \
        --model ${MODEL_PREFIX} \
        --mode ${BENCH_TYPE} \
        --bs ${BATCH_SIZE} \
        --cot ${DO_COT} \
        --max_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE} \
        --n_samples ${N_SAMPLES} \
        --base_url ${base_url} |& tee -a ${LOG_FILE}
  target_path=$(tail -n 1 "${LOG_FILE}")
  # Note that $() will suppress pdb, remove it if want to debug
  # target_path=$(python megatron_run.py \
  #       --root ${OUTPUT_PATH} \
  #       --model ${MODEL_PREFIX} \
  #       --mode ${BENCH_TYPE} \
  #       --bs ${BATCH_SIZE} \
  #       --cot ${DO_COT} \
  #       --max_tokens ${MAX_NEW_TOKENS} \
  #       --temperature ${TEMPERATURE} \
  #       --n_samples ${N_SAMPLES} \
  #       --base_url ${base_url} |& tee -a ${LOG_FILE} | tail -n 1)
  echo "Results output to: ${target_path}" |& tee -a ${LOG_FILE}
  key="${BENCH_TYPE}"
  target_paths["${key}"]="${target_path}"
  echo "---------------------------------------------" |& tee -a ${LOG_FILE}
done
cd ..


# --------------------------- Run the evaluation ---------------------------
echo -e "\n\n" |& tee -a ${LOG_FILE}
# #TODO: Shut down the Inference Server to free task slots, we don't need it at evaluation
# echo "Shutting down the inference server..." |& tee -a ${LOG_FILE}
# kill ${INFER_PID}

echo "[$(date)] Starting evaluation.." |& tee -a ${LOG_FILE}

if (( ${#target_paths[@]} > 2 )); then
    echo "Warning! More than 2 files to evaluate at once. Some have to wait since --ntasks-per-node=2" |& tee -a ${LOG_FILE}
fi
pids=()
for key in "${!target_paths[@]}"; do
  # Split the key into BENCH_TYPE and BENCH_SUBSET.
  target_path="${target_paths[$key]}"
  output_path=$(dirname "${target_path}")
  BENCH_TYPE=${key}
  echo "---------------------------------------------" |& tee -a ${LOG_FILE}
  echo "Evaluation file: ${target_path}" |& tee -a ${LOG_FILE}
  echo "Evaluating for bench_type: ${BENCH_TYPE}" |& tee -a ${LOG_FILE}

  # Build the evaluation command.
  cd evaluation
  python evaluate_generations.py \
      --generations_path ${target_path} \
      --scored_results_path "${output_path}/generations_scored.json" \
      --mode ${BENCH_TYPE} |& tee -a ${LOG_FILE}

  echo "---------------------------------------------" |& tee -a ${LOG_FILE}
done

echo "[$(date)] Evaluation completed for all setups." |& tee -a ${LOG_FILE}
