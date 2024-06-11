#!/bin/bash

# epochs=2
# epochs=10
epochs=50
# epochs=100

SPLIT=0

# --------------------------------
# 1. Uncomment to run within-study
# --------------------------------
# All train outputs are saved in params["model_outdir"]
# Uncomment to run within-study

SOURCE=gCSI
TARGET=$SOURCE


# -------------------------------
# 2. Uncomment to run cross-study
# -------------------------------
# All train outputs are saved in params["model_outdir"]
# Uncomment to run cross-study

# SOURCE=GDSCv1
# TARGET=CCLE


# ---------
# Run train
# ---------

python graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_outdir out_model/${SOURCE}/split_${SPLIT} \
    --epochs $epochs \
    --cuda_name cuda:7
