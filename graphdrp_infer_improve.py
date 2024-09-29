""" Inference with GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["output_dir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics. The scores are saved as json in
   test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specific imports
from model_params_def import infer_params # [Req]
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
)

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params: Dict) -> bool:
    """ Run model inference.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test") # [Req]

    # GraphDRP -- remove data_format
    test_data_fname = test_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTest data:")
    print(f"Infer_batch: {params['infer_batch']}")
    test_loader = build_GraphDRP_dataloader(
        data_dir=params["input_data_dir"],
        data_fname=test_data_fname,
        batch_size=params["infer_batch"],
        shuffle=False
    )

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])
    print(device)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    ) # [Req]
    model = load_GraphDRP(params, modelpath, device)
    model.eval()

    # Compute predictions
    test_true, test_pred = predicting(model, device, data_loader=test_loader)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    # [Req]
    additional_definitions = infer_params
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="graphdrp_params.txt",
        additional_definitions=additional_definitions
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
