Deep learning (DL) models built using popular DL frameworks can take various types of data from simple CSV to more complex structures such as `.pt` with PyTorch and `TFRecords` with TensorFlow.
Constructing datasets for drug response prediction (DRP) models generally requires combining heterogeneous data such as cancer and drug information and treatment response values.
We distinguish between two types of data:
- __ML data__. Data that can be directly consumed by prediction models for training and testing (e.g., `TFRecords`).
- __Raw data__. Data that are used to generate ML data (e.g., treatment response values, cancer and drug info). These usually include data files from drug sensitivity studies such as CCLE, CTRP, gCSI, GDSC, etc.

As part of model curation, the original data that is provided with public DRP models is copied to an FTP site. The full path is https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/ . For each model, a subdirectory is created for storing the model's data.

The raw data and ML data are located, respectively, in `data` and `data_processed` folders. E.g., the data for GraphDRP can be found in:
- https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data/
- https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data_processed/

Preprocessing scripts are often required to generate ML data from raw data. However, not all public repositories provide the necessary scripts.


# Raw data
The raw data is downloaded from GDSC website (version 6.0) and refers here to three types of data:
1) Dose-independent drug response values.
`PANCANCER_IC.csv`: drug and cell ids, IC50 values and other metadata (223 drugs and 948 cell lines).
2) Cancer sample information. `PANCANCER_Genetic_feature.csv`: 735 binary features that include coding variants and copy number alterations.
3) Drug information. `drug_smiles.csv`: SMILES strings of drug molecules. The SMILES were retrieved from PubChem using CIDs (Druglist.csv). The script `preprocess.py` provides functions to generate this file.

All these data types were provided with the GraphDRP repo. The data is available in: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data/


# ML data
The script `preprocess.py` uses raw data to generate ML data that can be used to train and test with GraphDRP. The necessary raw data are automatically downloaded from the FTP server using a `candle_lib` utility function `get_file()` and processed:

- __Response data__. IC50 values (PANCANCER_IC.csv) are transformed using 1 / (1 + pow(math.exp(float(ic50)), -0.1)).
- __Cancer features__. 735 binary features, including mutations and copy number alterations, are not modified.
- __Drug features__. SMILES string of each drug is converted into graph structure where nodes represent atoms and edges represent the bonds (each atom is represented by 78 features).

The user can specify one of three data splitting strategies: 1) mixed set (random split), 2) cell-blind (hard partition on cell line samples), 3) drug-blind (hard partition on drugs).
In either case, the script generates three files, `train_data.pt`, `val_data.pt`, and `test_data.pt` with 0.8/0.1/0.1 ratio, and saves them in appropriate directories:

- ./data_processed/<split_strategy>/processed/train_data.pt
- ./data_processed/<split_strategy>/processed/val_data.pt
- ./data_processed/<split_strategy>/processed/test_data.pt

Bash script `preprocess_batch.sh` generates the nine possible ML data configurations (i.e., 3 files for each splitting strategy). All the ML data files for the three splitting strategies are available in FTP: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data_processed/


# Using your own data
Ultimately, we want to be able to train models with other datasets (not only the ones provided with the model repo). This requires the preprocessing scripts to be available and reproducible.
