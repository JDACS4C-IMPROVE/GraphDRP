#!/bin/bash

# Below are two examples of how to run the end-to-end scripts:
# 1. Within-study analysis
# 2. Cross-study analysis

# Note! The outputs from preprocess, train, and infer are saved the same dir.

# ======================================================================
# To setup improve env vars, run this script first:
# source ./setup_improve.sh
# ======================================================================

# Download CSA data (if needed)
data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
fi

SPLIT=0

# EPOCHS=2
EPOCHS=150
CUDA_NAME=cuda:6

# ----------------------------------------
# 1. Within-study
# ---------------

SOURCE=CCLE
# SOURCE=gCSI
TARGET=$SOURCE

# Single dir
MLDATA_AND_MODEL_DIR=./res_same_dir/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess (improvelib)
python graphdrp_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $MLDATA_AND_MODEL_DIR

# Train (improvelib)
python graphdrp_train_improve.py \
    --input_dir $MLDATA_AND_MODEL_DIR \
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME \
    --epochs $EPOCHS

# Infer (improvelib)
python graphdrp_infer_improve.py \
    --input_data_dir $MLDATA_AND_MODEL_DIR\
    --input_model_dir $MLDATA_AND_MODEL_DIR\
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME


# ----------------------------------------
# 2. Cross-study
# --------------

SOURCE=GDSCv1
TARGET=CCLE

# Single dir
MLDATA_AND_MODEL_DIR=./res_same_dir/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess (improvelib)
python graphdrp_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $MLDATA_AND_MODEL_DIR

# Train (improvelib)
python graphdrp_train_improve.py \
    --input_dir $MLDATA_AND_MODEL_DIR \
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME \
    --epochs $EPOCHS

# Infer (improvelib)
python graphdrp_infer_improve.py \
    --input_data_dir $MLDATA_AND_MODEL_DIR\
    --input_model_dir $MLDATA_AND_MODEL_DIR\
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME
