#!/bin/bash

# Below are several examples of how to run the data train script.
# Currently, only CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

set -ue

# epochs=2
# epochs=10
# epochs=20
epochs=50
# epochs=100
# epochs=500

# Start timer
start_time=$(date +%s)

while true; do
    nvidia-smi --query-gpu=utilization.gpu --format=csv >> gpu_logs.txt;
    sleep 5;
done &


SPLIT=0


# ----------------------------------------
# Within-study
# ----------------------------------------

SOURCE=CCLE
TARGET=$SOURCE
# python graphdrp_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT} \
#     --epochs $epochs \
#     --cuda_name cuda:7
ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
python graphdrp_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $epochs \
    --cuda_name cuda:7

# ----------------------------------------
# Cross-study
# ----------------------------------------

# SOURCE=GDSCv1
# TARGET=CCLE
# # python graphdrp_train_improve.py \
# #     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
# #     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
# #     --model_outdir out_model/${SOURCE}/split_${SPLIT} \
# #     --cuda_name cuda:7
# #     # --epochs $epochs \
# ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
# MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
# python graphdrp_train_improve.py \
#     --input_dir $ML_DATA_DIR \
#     --output_dir $MODEL_DIR \
#     --epochs $epochs \
#     --cuda_name cuda:7


echo "GPU UTILIZATION"
awk 'NR%2==0 {print}' gpu_logs.txt | tr -d " %" > gpu_log_strip.txt
awk '{s+=$1}END{print "average %:",s/NR}' gpu_log_strip.txt
awk '{if($1>0+max){max=$1}} END{print "peak %:",max}' gpu_log_strip.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
