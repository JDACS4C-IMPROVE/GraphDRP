#!/bin/bash

# ----------------------------------------------
# Robustness analysis
# ----------------------------------------------

# This script abs path
# script_dir="$(dirname "$0")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "Script full path directory: $script_dir"

gout=${script_dir}/out.robust

# epochs=2
epochs=200
# cuda_name=cuda:4
# cuda_name=cuda:5
# cuda_name=cuda:6
cuda_name=cuda:7

reps=20
split=0
# src=CCLE
# src=CTRPv2
src=gCSI
# src=GDSCv1
# src=GDSCv2
trg=$src

start_rep=0

for r in $(seq $start_split 1 $reps); do

    # Separate dirs
    ml_data_dir=$gout/${src}_${split}/run_$r
    model_dir=$gout/${src}_${split}/run_$r
    infer_dir=$gout/${src}_${split}/run_$r

    # Preprocess (improvelib)
    python graphdrp_preprocess_improve.py \
        --train_split_file ${src}_split_${split}_train.txt \
        --val_split_file ${src}_split_${split}_val.txt \
        --test_split_file ${trg}_all.txt \
        --input_dir ./csa_data/raw_data \
        --output_dir $ml_data_dir

    # Train (improvelib)
    python graphdrp_train_improve.py \
        --input_dir $ml_data_dir \
        --output_dir $model_dir \
        --cuda_name $cuda_name \
        --epochs $epochs

    # Infer (improvelib)
    python graphdrp_infer_improve.py \
        --input_data_dir $ml_data_dir\
        --input_model_dir $model_dir\
        --output_dir $infer_dir \
        --calc_infer_score true \
        --cuda_name $cuda_name

done

# # Infer (improvelib)
# python graphdrp_infer_improve.py \
#     --input_data_dir $ml_data_dir\
#     --input_model_dir $model_dir\
#     --output_dir $infer_dir \
#     --calc_infer_score true \
#     --cuda_name $cuda_name

# ----------------------------------------
# SETS=(1 2 3)

# for SET in ${SETS[@]}; do
#     out_dir=$GLOBAL_SUFX/set${SET}
#     echo "Outdir $out_dir"
#     for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#         device=$(($split % 6))
#         echo "Set $SET; Split $split; Device $device"
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET \
#             exec >logs/run"$split".log 2>&1 &
#     done
# done

# ------------------------------------------------------------
# SET=1
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done

# SET=2
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done

# SET=3
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done
