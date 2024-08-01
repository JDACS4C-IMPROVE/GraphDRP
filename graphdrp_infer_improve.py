""" Inference with GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["infer_outdir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
# from improve import framework as frm
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specific imports
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
)

# [Req] Imports from preprocess and train scripts
from graphdrp_preprocess_improve import preprocess_params
from graphdrp_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# infer_params = app_infer_params + model_infer_params
infer_params = model_infer_params
# ---------------------


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    # frm.create_outdir(outdir=params["infer_outdir"]) # TODO cfg.initialize_parameters creates params['output_dir'] where the model will be stored

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # GraphDRP -- remove data_format
    test_data_fname = test_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTest data:")
    # print(f"test_ml_data_dir: {params['test_ml_data_dir']}")
    print(f"test_batch: {params['test_batch']}")
    if "input_data_dir" in params:
        data_dir = params["input_data_dir"]
    else:
        data_dir = params["input_dir"]
    test_loader = build_GraphDRP_dataloader(data_dir=data_dir,
                                            data_fname=test_data_fname,
                                            batch_size=params["test_batch"],
                                            shuffle=False)

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    if "input_model_dir" in params:
        model_dir = params["input_model_dir"]
    else:
        model_dir = params["input_dir"]
    modelpath = frm.build_model_path(params, model_dir=model_dir) # [Req]
    model = load_GraphDRP(params, modelpath, device)
    model.eval()

    # Compute predictions
    test_true, test_pred = predicting(model, device, data_loader=test_loader)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        # outdir=params["infer_outdir"]
        outdir=params["output_dir"] # TODO instead of infer_outdir
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        # outdir=params["infer_outdir"],
        outdir=params["output_dir"], # TODO instead of infer_outdir
        metrics=metrics_list
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    # params = frm.initialize_parameters(
    #     filepath,
    #     # default_model="graphdrp_default_model.txt",
    #     # default_model="graphdrp_params.txt",
    #     # default_model="params_ws.txt",
    #     # default_model="params_cs.txt",
    #     default_model="params_ovarian.txt",
    #     additional_definitions=additional_definitions,
    #     # required=req_infer_args,
    #     required=None,
    # )
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="graphdrp_params.txt",
        default_model=None,
        additional_cli_section=None,
        additional_definitions=additional_definitions,
        required=None)
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
