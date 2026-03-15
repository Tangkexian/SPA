#!/bin/bash

# --- Configuration --- #
MODEL_NAME="gpt-4o-mini" 
VLLM_SERVER_GPUS="0" # comma-separated list of GPU ids to use for vLLM server, e.g., "0,1,2,3"

# dataset parameters
DATASET_NAME="mhrag" # choices: "quality", "mhrag", "squad"
DATASET_IN="MultiHopRAG/corpus.json"
DATASET_OUT="data/synthetic_mhrag/mhrag.json" # the json path to save the generated data
NUM_ARTICLES=-1 # how many articles (-1 for all) 
START_ARTICLE=0 # start from this article number
K=100 # number of samples to generate per document

# Generation parameters
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
MIN_P=0
MAX_TOKENS=16384
PROMPT_KEY="key_concepts" # which prompt to use, choices: "implications", "key_concepts", "teacher_style", "discussions", "case_studies", "mind_map", "qa_critical_thinking"
INSTRUCT_MODEL=0 # choices: 0 (base model), 1 (instruct model)
USE_API=1 #choices: 0 (vLLM), 1 (API)

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