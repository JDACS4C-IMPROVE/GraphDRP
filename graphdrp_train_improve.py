""" Train GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["model_outdir"].

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
# from torch_geometric.data import DataLoader

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics
from candle import CandleCkptPyTorch

# Model-specific imports
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

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {"name": "model_arch",
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "type": str,
     "help": "Model architecture to run."},
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

def config_checkpointing(params: Dict, model, optimizer):
    """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

    Args:
        ckpt_directory (str): String with path to directory for storing the
            CANDLE checkpointing for the model being trained.

    Returns:
        Number of training iterations already run (this may be > 0 if reading
            from checkpointing).
    """
    # params["ckpt_directory"] = ckpt_directory
    initial_epoch = 0
    # TODO. This creates directory self.params["ckpt_directory"]
    # import pdb; pdb.set_trace()
    ckpt = CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
    return ckpt, initial_epoch


# [Req]
def run(params):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

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
    print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    print(f"batch_size: {params['batch_size']}")
    train_loader = build_GraphDRP_dataloader(params["train_ml_data_dir"],
                                             train_data_fname,
                                             params["batch_size"],
                                             shuffle=True)

    # Don't shuffle the val_loader, otherwise results will be corrupted
    print("\nVal data:")
    print(f"val_ml_data_dir: {params['val_ml_data_dir']}")
    print(f"val_batch: {params['val_batch']}")
    val_loader = build_GraphDRP_dataloader(params["val_ml_data_dir"],
                                           val_data_fname,
                                           params["val_batch"],
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
    # [Req] Set checkpointing
    print(f"model_outdir:   {params['model_outdir']}")
    print(f"ckpt_directory: {params['ckpt_directory']}")
    # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # params["output_dir"] = params["model_outdir"]
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]
        # params["ckpt_directory"] = "ckpt_graphdrp"  # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])
    ckpt_obj, initial_epoch = config_checkpointing(params, model, optimizer)

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
    print(f"Epochs: {initial_epoch} to {num_epoch}")
    for epoch in range(initial_epoch, num_epoch):
        # Train epoch and ckechpoint model
        train_loss = train_epoch(model, device, train_loader, optimizer, loss_fn, epoch + 1, log_interval)
        ckpt_obj.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

        # Predict with val data
        val_true, val_pred = predicting(model, device, val_loader)
        val_scores = compute_metrics(val_true, val_pred, metrics_list)

        # For early stop
        print(f"{early_stop_metric}, {val_scores[early_stop_metric]}")
        if val_scores[early_stop_metric] < best_score:
            torch.save(model.state_dict(), modelpath)
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            print(f"{early_stop_metric} improved at epoch {best_epoch};  \
                     Best {early_stop_metric}: {best_score};  Model: {params['model_arch']}")
            early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
        else:
            print(f"No improvement since epoch {best_epoch};  \
                     Best {early_stop_metric}: {best_score};  Model: {params['model_arch']}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == patience:
            print(f"Terminate training (model did not improve on val data for {params['patience']} epochs).")
            print(f"Best epoch: {best_epoch};  Best score ({early_stop_metric}): {best_score}")
            break

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
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores


# [Req]
def main(args):
# [Req]
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        # default_model="params_ws.txt",
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_train_args,
        required=None,
    )
    val_scores = run(params)
    print("\nFinished training GraphDRP model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
