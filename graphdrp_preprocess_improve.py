""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the
GraphDRP prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["output_dir"].

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
import pandas as pd
import joblib

# [Req] IMPROVE imports
# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm
# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specific imports
from model_params_def import preprocess_params # [Req]
from model_utils.torch_utils import TestbedDataset
from model_utils.utils import gene_selection, scale_df
from model_utils.rdkit_utils import build_graph_dict_from_smiles_collection
from model_utils.np_utils import compose_data_arrays

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    ## need to add

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data required by the model.
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
    ge = drp.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])
    #ge.reset_index()

    print("\nLoad drugs data.")
    smi = drp.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    
    print("Load train response data.")
    response_train = drp.get_response_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   response_file=params['y_data_file'])

    print("Find intersection of training data.")
    response_train = drp.get_response_with_features(response_train, ge, params['canc_col_name'])
    response_train = drp.get_response_with_features(response_train, smi, params['drug_col_name'])
    ge_train = drp.get_features_in_response(ge, response_train, params['canc_col_name'])

    print("Determine transformations.")
    drp.determine_transform(ge_train, 'ge_transform', params['cell_transcriptomic_transform'], params['output_dir'])




    # Prep molecular graph data for GraphDRP
    #smi = smi.reset_index()




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
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = drp.get_response_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                response_file=params['y_data_file'])
        response_stage = drp.get_response_with_features(response_stage, ge, params['canc_col_name'])
        response_stage = drp.get_response_with_features(response_stage, smi, params['drug_col_name'])
        ge_stage = drp.get_features_in_response(ge, response_stage, params['canc_col_name'])
        smi_stage = drp.get_features_in_response(smi, response_stage, params['drug_col_name'])

        print(f"Transform {stage} data.")
        ge_stage = drp.transform_data(ge_stage, 'ge_transform', params['output_dir'])
        # Prefix gene column names with "ge."
        fea_sep = "."
        fea_prefix = "ge"
        ge_stage = ge_stage.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge_stage.columns[1:]})

        if 'SMILES' in smi_stage.columns:
            smi_stage = smi_stage[['SMILES']]
        else:
            smi_stage.columns = ['SMILES']
        smiles_graphs = build_graph_dict_from_smiles_collection(smi_stage['SMILES'].values)


        # Sub-select desired response column (y_col_name)
        # ... and reduce response df to 3 columns: drug_id, cell_id and selected drug_response
        rsp_cut = response_stage[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]].copy()
        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(
            df_response=rsp_cut,
            df_drug=smi_stage,
            df_cell=ge_stage,
            drug_col_name=params["drug_col_name"],
            canc_col_name=params["canc_col_name"]
        )
        print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # -----------------------
        # [Req] Save ML data files in params["output_dir"]
        # The implementation of this step depends on the model.
        # -----------------------
        # [Req] Create data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        # Revmoe data_format because TestbedDataset() appends '.pt' to the
        # file name automatically. This is unique for GraphDRP.
        data_fname = data_fname.split(params["data_format"])[0]

        # Create the ml data and save it as data_fname in params["output_dir"]
        # Note! In the *train*.py and *infer*.py scripts, functionality should
        # be implemented to load the saved data.
        # -----
        # In GraphDRP, TestbedDataset() is used to create and save the file.
        # TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        # automatically creates dir called "processed" inside root and saves the file
        # inside. This results in: [root]/processed/[dataset],
        # e.g., ml_data/processed/train_data.pt
        # -----
        TestbedDataset(root=params["output_dir"],
                       dataset=data_fname,
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smiles_graphs)

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf=response_stage, stage=stage, output_dir=params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="graphdrp_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
