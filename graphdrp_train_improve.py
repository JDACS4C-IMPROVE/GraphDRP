""" Train GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["output_dir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For GraphDRP, the saved model:
        model.pt

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import torch

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params # [Req]
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
    set_GraphDRP,
    train_epoch,
)

# [Req] Imports from preprocess script
from graphdrp_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  


# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Build model path
    # ------------------------------------------------------
    modelpath = frm.build_model_path(params, model_dir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]

    # GraphDRP-specific -- remove data_format
    train_data_fname = train_data_fname.split(params["data_format"])[0]
    val_data_fname = val_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTrain data:")
    print(f"batch_size: {params['batch_size']}")
    sys.stdout.flush()
    train_loader = build_GraphDRP_dataloader(data_dir=params["input_dir"],
                                             data_fname=train_data_fname,
                                             batch_size=params["batch_size"],
                                             shuffle=True)

    # Don't shuffle the val_loader, otherwise results will be corrupted
    print("\nVal data:")
    print(f"val_batch: {params['val_batch']}")
    sys.stdout.flush()
    val_loader = build_GraphDRP_dataloader(data_dir=params["input_dir"],
                                           data_fname=val_data_fname,
                                           batch_size=params["val_batch"],
                                           shuffle=False)

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    # Model, Loss, Optimizer
    model = set_GraphDRP(params, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.MSELoss() # mse loss func

    # ------------------------------------------------------
    # Train settings
    # ------------------------------------------------------
    initial_epoch = 0
    num_epoch = params["epochs"]
    log_interval = params["log_interval"]
    patience = params["patience"]

    # Settings for early stop and best model settings
    best_score = np.inf
    best_epoch = -1
    early_stop_counter = 0  # define early-stop counter
    early_stop_metric = params["early_stop_metric"]  # metric to monitor for early stop

    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------
    epoch_list = []
    val_loss_list = []
    train_loss_list = []
    log_interval_epoch = 5

    print(f"Epochs: {initial_epoch + 1} to {num_epoch}")
    sys.stdout.flush()
    for epoch in range(initial_epoch, num_epoch):
        print(f"Start epoch: {epoch}")
        # Train epoch and checkpoint model
        train_loss = train_epoch(model, device, train_loader, optimizer,
                                 loss_fn, epoch + 1, log_interval)
        # ckpt_obj.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

        # Predict with val data
        val_true, val_pred = predicting(model, device, val_loader)
        val_scores = compute_metrics(val_true, val_pred, metrics_list)

        if epoch % log_interval_epoch == 0:
            epoch_list.append(epoch)
            val_loss_list.append(val_scores[early_stop_metric])

            train_true, train_pred = predicting(model, device, train_loader)
            train_scores = compute_metrics(train_true, train_pred, metrics_list)
            train_loss_list.append(train_scores[early_stop_metric])

        # For early stop
        print(f"{early_stop_metric}, {val_scores[early_stop_metric]}")
        sys.stdout.flush()
        if val_scores[early_stop_metric] < best_score:
            torch.save(model.state_dict(), modelpath)
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            print(f"{early_stop_metric} improved at epoch {best_epoch};  "\
                  f"Best {early_stop_metric}: {best_score};  "\
                  f"Model: {params['model_arch']}")
            early_stop_counter = 0  # reset early-stop counter if model improved after the epoch
        else:
            print(f"No improvement since epoch {best_epoch};  "\
                  f"Best {early_stop_metric}: {best_score};  "\
                  f"Model: {params['model_arch']}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == patience:
            print(f"Terminate training (model did not improve on val data for "\
                  f"{params['patience']} epochs).")
            print(f"Best epoch: {best_epoch};  Best score ({early_stop_metric}): {best_score}")
            break

    history = pd.DataFrame({"epoch": epoch_list,
                            "val_loss": val_loss_list,
                            "train_loss": train_loss_list})

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    model = load_GraphDRP(params, modelpath, device)
    model.eval()

    # Compute predictions
    val_true, val_pred = predicting(model, device, data_loader=val_loader)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["output_dir"],
        metrics=metrics_list
    )

    history.to_csv(Path(params["output_dir"])/"history.csv", index=False)

    return val_scores


# [Req]
def initialize_parameters():
    """This initialize_parameters() is define this way to support Supervisor
    workflows such as HPO.

    Returns:
        dict: dict of IMPROVE/CANDLE parameters and parsed values.
    """
    additional_definitions = train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="graphdrp_params.txt",
        default_model=None,
        additional_cli_section=None,
        additional_definitions=additional_definitions,
        required=None)
    return params


# [Req]
def main(args):
    # [Req]
    params = initialize_parameters()
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
