#!/bin/bash
export WANDB_MODE=offline
MODEL="meta-llama/Meta-Llama-3-8B"
MODEL_NAME="${MODEL##*/}"
EXPERIMENTS=(
    "TAG TOKENIZED_PATH 2 3e-5 2 4 0.0 0.03"
)
OUTPUT_DIR="results/cpt_fullft"
for EXP in "${EXPERIMENTS[@]}"; do
    read -r TAG TOKENIZED_PATH EPOCHS LR BS GA WD WR<<< "${EXP}"
    WANDB_PROJECT="general-knowledge-cpt"
    WANDB_RUN_NAME="${MODEL_NAME}_${TAG}_lr${LR}_bs${BS}_ga${GA}_ep${EPOCHS}_wd${WD}_wr${WR}"

    LOG_FILE="logs/${MODEL_NAME}_${TAG}_lr${LR}_bs${BS}_ga${GA}_ep${EPOCHS}_wd${WD}_wr${WR}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Run tag: ${TAG}" | tee -a "$LOG_FILE"

    torchrun --nproc_per_node=8 -m src.train.CPT_train \
        --tokenized_dataset_path "${TOKENIZED_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --model "${MODEL}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --batch_size "${BS}" \
        --weight_decay "${WD}" \
        --warmup_ratio "${WR}" \
        --gradient_accumulation_steps "${GA}" \
        --bf16 \
        --tag "${TAG}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        >> "${LOG_FILE}" 2>&1
done
