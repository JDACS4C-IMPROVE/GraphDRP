""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the
GraphDRP prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["ml_data_outdir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].
   For GraphDRP, the generated files:
        train_data.pt, val_data.pt, test_data.pt

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports
from model_utils.torch_utils import TestbedDataset
from model_utils.utils import gene_selection, scale_df
from model_utils.rdkit_utils import build_graph_dict_from_smiles_collection
from model_utils.np_utils import compose_data_arrays

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
# 
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file
app_preproc_params = [
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },

]

# 2. Model-specific params (Model: GraphDRP)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)  

    # Create output dir for model input data (to save preprocessed ML data)
    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data that is required by the model.
    #
    # Benchmark data includes three dirs: x_data, y_data, splits.
    # The x_data contains files that represent feature information such as
    # cancer representation (e.g., omics) and drug representation (e.g., SMILES).
    #
    # Prediction models utilize various types of feature representations.
    # Drug response prediction (DRP) models generally use omics and drug features.
    #
    # If the model uses omics data types that are provided as part of the benchmark
    # data, then the model must use the provided data loaders to load the data files
    # from the x_data dir.
    print("\nLoads omics data.")
    omics_obj = drp.OmicsLoader(params)
    # print(omics_obj)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

    print("\nLoad drugs data.")
    drugs_obj = drp.DrugsLoader(params)
    # print(drugs_obj)
    smi = drugs_obj.dfs['drug_SMILES.tsv']  # return SMILES data

    # ------------------------------------------------------
    # Further preprocess X data
    # ------------------------------------------------------
    # Gene selection (based on LINCS landmark genes)
    if params["use_lincs"]:
        genes_fpath = filepath/"model_utils/landmark_genes.txt"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # Prefix gene column names with "ge."
    fea_sep = "."
    fea_prefix = "ge"
    ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

    # Prep molecular graph data for GraphDRP
    smi = smi.reset_index()
    smi.columns = [params["drug_col_name"], "SMILES"]
    drug_smiles = smi["SMILES"].values  # list of smiles
    smiles_graphs = build_graph_dict_from_smiles_collection(drug_smiles)

    # ------------------------------------------------------
    # Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    print("Create feature scaler.")
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge( ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(smi[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
    # smi_sub = smi[smi[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)
    # TODO: consider keeping only smiles (smiles_graphs) in smi_sub

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge( ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(smi[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
        smi_sub = smi[smi[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)
        # print(rsp[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        # Scale features
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler) # scale gene expression
        # print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        # print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())

        # Sub-select desired response column (y_col_name)
        # ... and reduce response df to 3 columns: drug_id, cell_id and selected drug_response
        rsp_cut = rsp[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]].copy()
        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(
            rsp_cut, smi, ge_sc, params["drug_col_name"], params["canc_col_name"]
        )
        print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # -----------------------
        # [Req] Create data name
        data_fname = frm.build_ml_data_name(params, stage)

        # Revmoe data_format because TestbedDataset() appends '.pt' to the
        # file name automatically. This is unique for GraphDRP.
        data_fname = data_fname.split(params["data_format"])[0]

        # Create the ml data and save it as data_fname in params["ml_data_outdir"]
        # Note! In the *train*.py and *infer*.py scripts, functionality should
        # be implemented to load the saved data.
        # -----
        # In GraphDRP, TestbedDataset() is used to create and save the file.
        # TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        # automatically creates dir called "processed" inside root and saves the file
        # inside. This results in: [root]/processed/[dataset],
        # e.g., ml_data/processed/train_data.pt
        # -----
        TestbedDataset(root=params["ml_data_outdir"],
                       dataset=data_fname,
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smiles_graphs)

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(rsp, params, stage)

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        default_model="params_ws.txt",
        # default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
