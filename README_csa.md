
# Step-by-step instructions to run cross study analysis using PARSL on Lambda machine

### 1. Clone the GraphDRP repo
```
git clone https://github.com/JDACS4C-IMPROVE/GraphDRP.git
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
Create conda env using `env_gdrp_37_improve.yml`
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
csa_params.ini contains parameters necessary for the workflow. The user can change the parameters inside this configuration file.

 - input_dir : Location of raw data for cross study analysis. 
 - output_dir : Location of the inference results
 - source_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - target_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - split: Splits of the source datasets for cross study analysis.
 - model_name: Name of the model for cross study analysis
 - epochs: Number of epochs for the model
 - y_col_name: Response variable used in the model. eg: auc
 - use_singularity: True, if the model files are available in a singularity container
 - only_cross_study: True, if only cross study analysis is needed without within study inferences


 To run cross study analysis with default configuration file (csa_params.ini):
```
python workflow_csa.py
```
 To run cross study analysis with a different configuration file:
```
python workflow_csa.py --config_file <CONFIG_FILE>
```