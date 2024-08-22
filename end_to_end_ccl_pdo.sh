#!/bin/bash

current_dir=$PWD
echo "PWD: $current_dir"

MAINDIR=results_ccl_pdo

# Preprocess
python graphdrp_preprocess_improve.py \
    --train_split_file combind_ccl_pdo_trainID.txt \
    --val_split_file combind_ccl_pdo_valID.txt \
    --test_split_file combind_ccl_pdo_testID.txt \
    --ml_data_outdir results_ccl_pdo

# Train
python graphdrp_train_improve.py \
    --train_ml_data_dir results_ccl_pdo \
    --val_ml_data_dir results_ccl_pdo \
    --model_outdir results_ccl_pdo \
    --cuda_name "cuda:7"

# Infer
python graphdrp_infer_improve.py \
    --test_ml_data_dir results_ccl_pdo \
    --model_dir results_ccl_pdo \
    --infer_outdir results_ccl_pdo \
    --cuda_name "cuda:7"
