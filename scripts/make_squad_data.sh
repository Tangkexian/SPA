#!/bin/bash

# --- Model Configuration --- #
MODEL_NAME="Qwen/Qwen2.5-7B" 
VLLM_SERVER_GPUS="0,1"  # GPUs for vLLM server (e.g. "0,1,2,3")

# --- Dataset Configuration --- #
DATASET_NAME="squad"      # choices: quality, mhrag, squad
DATASET_IN="data/SQuAD/squad_val_200.json"
PROMPT_KEY="implications" # prompt type:
# implications | key_concepts | teacher_style | discussions
# case_studies | mind_map | qa_critical_thinking
DATASET_OUT="data/synthetic_squad/squad_${PROMPT_KEY}.json" 
NUM_ARTICLES=3   # number of articles to process (-1 for all)
START_ARTICLE=0  # starting article index
K=10             # number of samples generated per document

# --- Generation Parameters --- #
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
MIN_P=0
MAX_TOKENS=8192

# --- Inference Backend --- #
INSTRUCT_MODEL=0    # 0: base model | 1: instruct model
USE_API=0           # 0: vLLM | 1: API

# --- Derived Parameters --- #
TP_SIZE=$(echo "${VLLM_SERVER_GPUS}" | awk -F',' '{print NF}')
INS_FLAG=""
if [[ "${INSTRUCT_MODEL}" == "1" ]]; then
    INS_FLAG="--instruct_model"
fi
API_FLAG=""
if [[ "${USE_API}" == "1" ]]; then
    API_FLAG="--use_api"
fi

# --------------------------------------------------------------------- #
echo "Running offline vLLM data generation on GPUs ${VLLM_SERVER_GPUS} (TP=${TP_SIZE})"
CUDA_VISIBLE_DEVICES=${VLLM_SERVER_GPUS} python3 -m src.generate_data \
    --dataset_name "$DATASET_NAME" \
    --model "$MODEL_NAME" \
    --tensor_parallel_size "${TP_SIZE}" \
    --dataset_in "$DATASET_IN" \
    --dataset_out "$DATASET_OUT" \
    --n "$NUM_ARTICLES" \
    --start "$START_ARTICLE" \
    --k "$K" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --min_p "$MIN_P" \
    --max_tokens "$MAX_TOKENS" \
    --prompt_key "$PROMPT_KEY" \
    ${INS_FLAG} \
    ${API_FLAG}
echo "Job finished."