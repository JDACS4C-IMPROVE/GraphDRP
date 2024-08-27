# GraphDRP

This repository demonstrates how to use the [IMPROVE library v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta) for building a drug response prediction (DRP) model using GraphDRP, and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.0.3-beta`, is the final release before transitioning to `v0.1.0-alpha`, which introduces a new API. Version `v0.0.3-beta` and all previous releases have served as the foundation for developing essential components of the IMPROVE software stack. Subsequent releases build on this legacy with an updated API, designed to encourage broader adoption of IMPROVE and its curated models by the research community.

A more detailed tutorial can be found [here](https://jdacs4c-improve.github.io/docs/content/unified_interface.html). 
`TODO`: update with the new docs!


## Dependencies
Installation instuctions are detialed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `yml` file [conda_env_py37.sh](./conda_env_py37.sh)

ML framework:
+ [Torch](https://pytorch.org/) -- deep learning framework for building the prediction model
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) -- graph neural networks (GNN)

IMPROVE dependencies:
+ [IMPROVE v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta)
+ [candle_lib](https://github.com/ECP-CANDLE/candle_lib) - IMPROVE dependency (enables various hyperparameter optimization on HPC machines) `TODO`: need to fork into IMPROVE project and tag



## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```

Note that `./_original_data` contains data files that were used to train and evaluate the GraphDRP for the original paper.


## Model scripts and parameter file
+ `graphdrp_preprocess_improve.py` - takes benchmark data files and transforms into files for trianing and inference
+ `graphdrp_train_improve.py` - trains the GraphDRP model
+ `graphdrp_infer_improve.py` - runs inference with the trained GraphDRP model
+ `graphdrp_params.txt` - default parameter file



# Step-by-step instructions

### 1. Clone the model repository
```
git clone git@github.com:JDACS4C-IMPROVE/GraphDRP.git
cd GraphDRP
git checkout develop
```


### 2. Set computational environment
Option 1: create conda env using `yml`
```
conda env create -f conda_env_lambda_graphdrp_py37.yml
```

Option 2: check [conda_env_py37.sh](./conda_env_py37.sh)


### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout tag `v0.0.3-beta`) outside the GraphDRP model repo
3. Set up env variables: `IMPROVE_DATA_DIR` (to `./csa_data/`) and `PYTHONPATH` (adds IMPROVE repo).


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python graphdrp_preprocess_improve.py
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `train_data.pt`, `val_data.pt`, `test_data.pt`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
ml_data
└── GDSCv1-CCLE
    └── split_0
        ├── processed
        │   ├── test_data.pt
        │   ├── train_data.pt
        │   └── val_data.pt
        ├── test_y_data.csv
        ├── train_y_data.csv
        ├── val_y_data.csv
        └── x_data_gene_expression_scaler.gz
```


### 5. Train GraphDRP model
```bash
python graphdrp_train_improve.py
```

Trains GraphDRP using the model input data: `train_data.pt` (training), `val_data.pt` (for early stopping).

Generates:
* trained model: `model.pt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── GDSCv1
    └── split_0
        ├── best -> /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP/out_models/GDSCv1/split_0/epochs/002
        ├── epochs
        │   ├── 001
        │   │   ├── ckpt-info.json
        │   │   └── model.h5
        │   └── 002
        │       ├── ckpt-info.json
        │       └── model.h5
        ├── last -> /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP/out_models/GDSCv1/split_0/epochs/002
        ├── model.pt
        ├── out_models
        │   └── GDSCv1
        │       └── split_0
        │           └── ckpt.log
        ├── val_scores.json
        └── val_y_data_predicted.csv
```


### 6. Run inference on test data with the trained model
```python graphdrp_infer_improve.py```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── GDSCv1-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
