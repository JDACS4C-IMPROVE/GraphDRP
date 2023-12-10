GraphDRP model for drug response prediction (DRP). The use of this model is demonstrated here with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

# Dependencies
Check `conda_env.sh`
+ [Candle-lib](https://github.com/ECP-CANDLE/candle_lib) -- improve lib dependency
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)
+ [Rdkit](https://www.rdkit.org/)

## Source codes
+ `graphdrp_preprocess_improve.py`: creates data files for drug resposne prediction (DRP)
+ `graphdrp_train_improve.py`: trains a GraphDRP model
+ `graphdrp_infer_improve.py`: runs inference with the trained GraphDRP model
+ `graphdrp_params.txt`: parameter file

# Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).
The required data tree is shown below:

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

# Step-by-step running

## 1. Set the required computational environment
* Download benchmark data
```
sh ./download_csa.sh
```
* Install dependencies (check `conda_env.sh`)
* Set the required environment variables to point towards the data folder and improve lib.
Follow this repo to set up the env variables for `IMPROVE_DATA_DIR` and improve lib.
https://github.com/JDACS4C-IMPROVE/IMPROVE

## 2. Preprocess raw benchmark data to construct model input data
```bash
python preprocess_preprocess_improve.py
```
or 
```bash
sh preprocess_example.sh
```
This generates:
* three model input data files: `train_data.pt`, `val_data.pt`, `infer_data.pt`
* three data files, each containing y data (responses) and metadata: `train_y_data.csv`, `val_y_data.csv`, `infer_y_data.csv`

```
ml_data/
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

## 3. Train the GraphDRP model
```bash
python preprocess_train_improve.py
```

This trains GraphDRP using the processed data: `train_data.pt` (training), `val_data.pt` (early stopping).

This generates:
* trained model: `model.pt`
* predictions on val data: `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
* checkpointing results
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

## 4. Run the trained model in inference on test data
```python preprocess_infer_improve.py```

The script uses processed data and the trained model to evaluate performance.

This generates:
* predictions on test data: `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── GDSCv1-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
