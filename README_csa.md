# GraphDRP
GraphDRP model for drug response prediction (DRP).


# Dependencies
Create conda env using `env_gdrp_37_improve.yml`, or check [conda_env_py37.sh](./conda_env_py37.sh)
```
conda env create -f env_gdrp_37_improve.yml
```
Activate environment:
```
conda activate graphdrp_py37_improve
```

### Install Parsl (2023.6.19):
```
pip install parsl 
```
If you see an error during execution you may have to do this:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```


ML framework:
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) -- for graph neural network (GNN)

IMPROVE lib:
+ [improve_lib](https://github.com/JDACS4C-IMPROVE/IMPROVE)
+ [candle_lib](https://github.com/ECP-CANDLE/candle_lib) -- improve lib dependency


# Step-by-step running

### 1. Clone the GraphDRP repo
```
git clone https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop
cd GraphDRP
git checkout framework-api
```

### 2. Clone IMPROVE repo
Clone the `IMPROVE library` repository to a directory of your preference (outside of your drug response prediction (DRP) model's directory).

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout framework-api
```

### 3. Install Dependencies
Create conda env using `env_gdrp_37_improve.yml`, or check [conda_env_py37.sh](./conda_env_py37.sh)
```
conda env create -f env_gdrp_37_improve.yml
```
Activate environment:
```
conda activate graphdrp_py37_improve
```

Install Parsl (2023.6.19):
```
pip install parsl 
```
If you see an error during execution you may have to do this:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

### 4. Set PYTHONPATH and download benchmark data
```
source setup_improve.sh
```
This will set up `PYTHONPATH` to point the IMPROVE repo, and download cross-study benchmark data into `./csa_data/`.

## To run cross study analysus using PARSL on Lambda machine:
csa_params.ini contains parameters necessary for the workflow. However, please change the source_datasets, target_datasets, split, epochs within workflow_csa.py script. Run this for cross study analysis:
```
python workflow_csa.py
```