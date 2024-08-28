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
# from improve import framework as frm
# from improve.metrics import compute_metrics
# from candle import CandleCkptPyTorch
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

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
# Model-specific params (Model: GraphDRP)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {"name": "model_arch",
     "type": str,
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "help": "Model architecture to run."
    },
    {"name": "log_interval",
     "type": int,
     # "action": "store",
     "default": 20,
     "help": "Interval for saving o/p"
    },
    {"name": "cuda_name",
     "type": str,
     # "action": "store",
     "default": "cuda:0",
     "help": "Cuda device (e.g.: cuda:0, cuda:1)."
    },
    # TODO "learning_rate" is already defined in improvelib, but we can still
    #   define it in  *train*.py and doesn't throw error or warning!
    # {"name": "learning_rate",
    #  "type": float,
    #  "default": 0.0001,
    #  "help": "Learning rate for the optimizer."
    # },
]

# train_params = app_train_params + model_train_params
train_params = model_train_params
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
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    # frm.create_outdir(outdir=params["model_outdir"]) # TODO cfg.initialize_parameters creates params['output_dir'] where the model will be stored

    # Build model path
    # modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["output_dir"]) # TODO instead of model_outdir

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
    # print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    print(f"batch_size: {params['batch_size']}")
    sys.stdout.flush()
    # train_loader = build_GraphDRP_dataloader(params["train_ml_data_dir"],
    train_loader = build_GraphDRP_dataloader(data_dir=params["input_dir"],
                                             data_fname=train_data_fname,
                                             batch_size=params["batch_size"],
                                             shuffle=True)

    # Don't shuffle the val_loader, otherwise results will be corrupted
    print("\nVal data:")
    # print(f"val_ml_data_dir: {params['val_ml_data_dir']}")
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
    # # [Req] Set checkpointing
    # print(f"model_outdir:   {params['model_outdir']}")
    # print(f"ckpt_directory: {params['ckpt_directory']}")
    # # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # # params["output_dir"] = params["model_outdir"]
    # if params["ckpt_directory"] is None:
    #     params["ckpt_directory"] = params["model_outdir"]
    #     # params["ckpt_directory"] = "ckpt_graphdrp"  # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # # ckpt_obj, initial_epoch = config_checkpointing(params, model, optimizer)
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
    # log_interval_epoch = 1
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
            early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
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
        # outdir=params["model_outdir"]
        outdir=params["output_dir"] # TODO explore input_dir and output_dir
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        # outdir=params["model_outdir"],
        outdir=params["output_dir"], # TODO explore input_dir and output_dir
        metrics=metrics_list
    )

    history.to_csv(Path(params["output_dir"])/"history.csv", index=False)

    return val_scores


def initialize_parameters():
    # params = frm.initialize_parameters(
    #     filepath,
    #     # default_model="graphdrp_default_model.txt",
    #     # default_model="graphdrp_params.txt",
    #     # default_model="params_ws.txt",
    #     # default_model="params_cs.txt",
    #     default_model="params_ovarian.txt",
    #     additional_definitions=additional_definitions,
    #     # required=req_train_args,
    #     required=None,
    # )
    cfg = DRPTrainConfig()
    #additional_definitions = preprocess_params + train_params
    additional_definitions = train_params
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
    print("\nFinished training GraphDRP model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
