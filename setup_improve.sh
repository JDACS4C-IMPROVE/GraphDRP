#!/bin/bash --login

# This script
# Download csa data
# Sets up env vars: IMPROVE_DATA_DIR and PYTHONPATH

# run it like this: source ./setup_improve.sh
# TODO. Wierd behavior: only when this script is sourced, the tmux pane is automatically terminated
#       when I get an error in other bash scripts.

# set -e

# Get modelpath and modeldir
model_path=$PWD
echo "Model path: $model_path"
model_name=$(echo "$model_path" | awk -F '/' '{print $NF}')
echo "Model name: $model_name"

# Download data (if needed)
data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
else
    echo "CSA data folder already exists"
fi

# Set env var IMPROVE_DATA_DIR
export IMPROVE_DATA_DIR="./$data_dir/"

# Clone IMPROVE lib
# echo "Clone IMPROVE lib"
pushd ../
# git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
git clone git@github.com:JDACS4C-IMPROVE/IMPROVE.git
improve_lib_path=$PWD/IMPROVE
pushd $model_name

# Set env var PYTHOPATH
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
echo "PYTHONPATH: $PYTHONPATH"
