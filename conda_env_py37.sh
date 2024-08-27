#!/bin/bash --login

set -e

# Run this on Lambda
# source /etc/profile.d/lmod.sh
# which nvcc
# module avail
# module load cuda/11.8
# which nvcc

# CONDA_ENV_NAME=graphdrp_py37 # Conda env name
# conda create -n $CONDA_ENV_NAME python=3.7 pip --yes # Create conda env
# conda activate $CONDA_ENV_NAME # Activate conda env

# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
conda install pyg=2.1.0 -c pyg -c conda-forge --yes

conda install -c conda-forge matplotlib --yes # installed with candle
conda install -c conda-forge h5py=3.1.0 --yes

conda install -c bioconda pubchempy=1.0.4 --yes
conda install -c rdkit rdkit=2020.09.1.0 --yes # also installs pandas
conda install -c anaconda networkx=2.6.3 --yes
# conda install -c conda-forge pyarrow=10.0 --yes
# conda install -c conda-forge pyarrow=8.0.0 --yes
conda install conda-forge::pyarrow  # potential issues!?

conda install -c pyston psutil --yes

# Install CANDLE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

# # Other
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes

# Check installs
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"
# python -c "import networkx; print(networkx.__version__)"
# python -c "import matplotlib; print(matplotlib.__version__)"
# python -c "import h5py; print(h5py.version.info)"
# python -c "import pubchempy; print(pubchempy.__version__)"
# python -c "import rdkit; print(rdkit.__version__)"
