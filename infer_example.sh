#!/bin/bash

# # Cross-study
# python graphdrp_infer_improve.py \
#     --test_ml_data_dir ml_data/CCLE-CCLE/split_0 \
#     --model_dir out_model/CCLE/split_0 \
#     --infer_outdir out_infer/CCLE-CCLE/split_0 \
#     --cuda_name cuda:7

# Within-study
python graphdrp_infer_improve.py \
    --test_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_dir out_model/GDSCv1/split_0 \
    --infer_outdir out_infer/GDSCv1-CCLE/split_0
    --cuda_name cuda:7
