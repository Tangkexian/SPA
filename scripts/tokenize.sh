#!/bin/bash
MODEL="models/meta-llama/Meta-Llama-3-8B"
MODEL_NAME="${MODEL##*/}"

DATASET="data/synthetic_squad/squad_implications.json,data/synthetic_squad/squad_key_concepts.json,data/synthetic_squad/squad_teacher_style.json,data/synthetic_squad/squad_discussions.json,data/synthetic_squad/squad_case_studies.json,data/synthetic_squad/squad_mind_map.json,data/synthetic_squad/squad_qa_critical_thinking.json" # the dataset paths, separated by commas
DATASET_ARGS=${DATASET//,/ }
K_COMP=50,100,60,70,80,110,90 # the number of samples needed for each dataset, separated by commas
K_COMP_ARGS=${K_COMP//,/ }
K_COMP_TAG=${K_COMP//,/_}  
N_ARTICLES=-1

OUTPUT_DIR="data_tokenized"
if [ -z "$OUTPUT_DIR" ]; then
    echo "OUTPUT_DIR is not set"
    exit 1
fi
TAG="${MODEL_NAME}_k${K_COMP_TAG}"
LOG_FILE=${OUTPUT_DIR}/${TAG}/tokenize.log
mkdir -p ${OUTPUT_DIR}/${TAG}

python -u -m src.train.CPT_tokenize \
    --dataset ${DATASET_ARGS} \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --tag "${TAG}" \
    --k_completions ${K_COMP_ARGS} \
    --n_articles ${N_ARTICLES} \
    > ${LOG_FILE} 2>&1
