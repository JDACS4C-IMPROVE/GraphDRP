#!/bin/bash

current_dir=$PWD
echo "PWD: $current_dir"

# ----------------------------------------
# 1. Within-study
# ---------------

SPLIT=0

# SOURCE=CTRPv2
SOURCE=GDSCv2

# -----------------------
# Single-dataset analysis
# -----------------------

# TARGET=$SOURCE

# # Preprocess
# # All preprocess outputs are saved in params["ml_data_outdir"]
# python graphdrp_preprocess_improve.py \
#     --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
#     --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
#     --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
#     --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

# # Train
# # All train outputs are saved in params["model_outdir"]
# python graphdrp_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT}

# # Infer
# # All infer outputs are saved in params["infer_outdir"]
# python graphdrp_infer_improve.py \
#     --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_dir out_model/${SOURCE}/split_${SPLIT} \
#     --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT}


# -----------------------
# PDMR-dataset analysis
# -----------------------

TARGET=PDMR

# Preprocess
# All preprocess outputs are saved in params["ml_data_outdir"]
python graphdrp_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

# Train
# All train outputs are saved in params["model_outdir"]
python graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_outdir out_model/${SOURCE}/split_${SPLIT}

# Infer
# All infer outputs are saved in params["infer_outdir"]
python graphdrp_infer_improve.py \
    --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_dir out_model/${SOURCE}/split_${SPLIT} \
    --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT}
