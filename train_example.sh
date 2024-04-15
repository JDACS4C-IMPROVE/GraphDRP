#!/bin/bash

epochs=2
# epochs=10
# epochs=20
# epochs=100

# Within-study
# All train outputs are saved in params["model_outdir"]
# SOURCE=CCLE
SOURCE=gCSI
TARGET=$SOURCE
SPLIT=1
python -m pdb graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_outdir out_model/${SOURCE}/split_${SPLIT} \
    --epochs $epochs \
    --cuda_name cuda:7


# # Cross-study
# # All train outputs are saved in params["model_outdir"]
# SOURCE=GDSCv1
# TARGET=CCLE
# SPLIT=0
# python graphdrp_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT} \
#     --cuda_name cuda:7
#     # --epochs $epochs \
