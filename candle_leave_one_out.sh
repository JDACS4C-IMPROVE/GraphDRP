#!/bin/bash

# Note! The outputs from preprocess, train, and infer are saved into different dirs.

# Line 162 in graphdrp_train_improve.py
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x3968 and 4096x128)

# ======================================================================
# To setup improve env vars, run this script first:
# source ./setup_improve.sh
# ======================================================================

DATA_DIR="repurposing_data"

# Params
MODEL_NAME=graphdrp

SPLIT=0

# EPOCHS=2
# EPOCHS=10
# EPOCHS=25
EPOCHS=50
# EPOCHS=500

CUDA=7

# This script abs path
# script_dir="$(dirname "$0")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "Script full path directory: $script_dir"

# Separate dirs
gout=${script_dir}/out.end_to_end_repurposing
ML_DATA_DIR=$gout/preprocess/split_${SPLIT}
MODEL_DIR=$gout/train/split_${SPLIT}
INFER_DIR=$gout/infer/split_${SPLIT}

# Preprocess
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_preprocess_improve.py \
    --train_split_file ${SPLIT}_train.txt \
    --val_split_file ${SPLIT}_val.txt \
    --test_split_file ${SPLIT}_test.txt \
    --input_dir ./${DATA_DIR}/raw_data \
    --output_dir $ML_DATA_DIR

# Train
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $EPOCHS

# Infer
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR \
    --calc_infer_score false
    # --calc_infer_score true
