#!/bin/bash

# Two examples of how to run the end-to-end scripts.
# Uncomment and run the one you are interested in.
# 1. Within-study analysis
# 2. Cross-study analysis


# # Download the benchmark CSA data
# wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/

# ======================================================================
# Set env variables:
# 1. IMPROVE_DATA_DIR
# 2. IMPROVE lib
# -------------------
current_dir=/lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP
echo "PWD: $current_dir"

# Set env variable for IMPROVE lib
# export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE

# Set env variable for IMPROVE_DATA_DIR (benchmark dataset)
# export IMPROVE_DATA_DIR="./csa_data/"
# echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
# ======================================================================

SPLIT=0

# --------------------------------
# 1. Uncomment to run within-study
# --------------------------------

# SOURCE=gCSI
# TARGET=$SOURCE


# -------------------------------
# 2. Uncomment to run cross-study
# -------------------------------

SOURCE=GDSCv1
TARGET=CCLE


# ------------------------------------
# Run pipeline (preproc, train, infer)
# ------------------------------------

# Preprocess
# All preprocess outputs are saved in params["ml_data_outdir"]
python graphdrp_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
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
