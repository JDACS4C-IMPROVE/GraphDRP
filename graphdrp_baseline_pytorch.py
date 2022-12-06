import argparse
import datetime
import os
from pathlib import Path
import sys
from random import shuffle
from time import time
import json

import candle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import graphdrp as bmk
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *


class Timer:
    """
    Measure runtime.
    """

    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        time_diff = self.end - self.start
        return time_diff

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff) // 3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format((time_diff) / 3600))
        else:
            print_fn("Runtime: {:.1f} mins".format((time_diff) / 60))


def train(model, device, train_loader, optimizer, epoch, log_interval):
    """ Training function at each epoch. """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data.x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def launch(modeling, args):

    timer = Timer()
    if args.set == "mixed":
        set_str = "_mixed"
        val_scheme = "mixed_set"
    elif args.set == "cell":
        set_str = "_cell_blind"
        val_scheme = "cell_blind"
    elif args.set == "drug":
        set_str = "_blind"
        val_scheme = "drug_blind"

    # Create output dir
    if args.output_dir is not None:
        outdir = Path(args.output_dir)
    else:
        outdir = fdir / "results"
    os.makedirs(outdir, exist_ok=True)

    # Fetch data (if needed)
    ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data_processed/{val_scheme}/processed"
    data_file_list = ["train_data.pt", "val_data.pt", "test_data.pt"]

    for f in data_file_list:
        candle.get_file(fname=f,
                        origin=os.path.join(ftp_origin, f.strip()),
                        unpack=False, md5_hash=None,
                        cache_subdir=args.cache_subdir )

    _data_dir = os.path.split(args.cache_subdir)[0]
    root = os.getenv('CANDLE_DATA_DIR') + '/' + _data_dir

    # CANDLE known params
    lr = args.learning_rate
    num_epoch = args.epochs
    train_batch = args.batch_size

    # Model specific params
    log_interval = args.log_interval
    val_batch = args.val_batch
    test_batch = args.test_batch

    print("Learning rate: ", lr)
    print("Epochs: ", num_epoch)

    model_st = modeling.__name__
    dataset = "GDSC"
    train_losses = []
    val_losses = []
    val_pearsons = []
    # print("\nrunning on ", model_st + "_" + dataset)

    # Prepare data loaders
    print("root: {}".format(root))
    file_train = args.train_data
    file_val = args.test_data
    file_test = args.test_data
    train_data = TestbedDataset(root=root, dataset=file_train)
    val_data = TestbedDataset(root=root, dataset=file_val)
    test_data = TestbedDataset(root=root, dataset=file_test)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    print("CPU/GPU: ", torch.cuda.is_available())

    # CUDA device from env var
    # assert os.getenv("CUDA_VISIBLE_DEVICES").isnumeric(), print("CUDA_VISIBLE_DEVICES must be numeric.")
    # cuda_name = f"cuda:{int(os.getenv('CUDA_VISIBLE_DEVICES'))}"
    # if os.getenv("CUDA_VISIBLE_DEVICES").isnumeric():
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        cuda_name = "cuda:0"
    else:
        cuda_name = args.cuda_name

    # Training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = modeling().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = 1000
    best_pearson = 1
    best_epoch = -1

    model_file_name = outdir / ("model_" + model_st + "_" + dataset + "_" + val_scheme + ".model")
    result_file_name = outdir / ("result_" + model_st + "_" + dataset + "_" + val_scheme + ".csv")
    loss_fig_name = str(outdir / ("model_" + model_st + "_" + dataset + "_" + val_scheme + "_loss"))
    pearson_fig_name = str(outdir / ("model_" + model_st + "_" + dataset + "_" + val_scheme + "_pearson"))

    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

        # Val set scores
        G, P = predicting(model, device, val_loader)
        ret = [rmse(G, P),
               mse(G, P),
               pearson(G, P),
               spearman(G, P)
        ]

        # Test set scores
        G_test, P_test = predicting(model, device, test_loader)
        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
        ]

        train_losses.append(train_loss)
        val_losses.append(ret[1])
        val_pearsons.append(ret[2])

        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, "w") as f:
                f.write(",".join(map(str, ret_test)))
            best_epoch = epoch + 1
            best_rmse = ret[0]
            best_mse = ret[1]
            best_pearson = ret[2]
            best_spearman = ret[3]
            print(f"RMSE improved at epoch {best_epoch}; Best RMSE: {best_mse}; Model: {model_st}; Dataset: {dataset}")
        else:
            print(f"No improvement since epoch {best_epoch}; Best RMSE: {best_mse}; Model: {model_st}; Dataset: {dataset}")

    draw_loss(train_losses, val_losses, loss_fig_name)
    draw_pearson(val_pearsons, pearson_fig_name)

    # Test set raw predictions
    G_test, P_test = predicting(model, device, test_loader)
    preds = pd.DataFrame({"True": G_test, "Pred": P_test})
    preds_file_name = f"test_preds_{val_scheme}_{model_st}_{dataset}.csv"
    preds.to_csv(outdir / preds_file_name, index=False)

    # Test set scores
    pcc_test = pearson(G_test, P_test)
    scc_test = spearman(G_test, P_test)
    rmse_test = rmse(G_test, P_test)
    test_scores = {"pcc": pcc_test, "scc": scc_test, "rmse": rmse_test}

    with open(outdir / f"test_scores_{val_scheme}_{model_st}_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    # Supervisor HPO
    val_scores = {"val_loss": float(best_mse), "pcc": float(best_pearson), "scc": float(best_spearman), "rmse": float(best_rmse)}

    timer.display_timer()
    print("Scores:\n\t{}".format(val_scores))
    
    return val_scores


def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]

    # Call launch() with specific model arch and args with all HPs
    scores = launch(modeling, args)
    
    # Dump values for Supervisor HPO
    import json
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores['rmse']))
    with open(args.output_dir + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)
    
    return scores


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    graphdrp_bmk = bmk.BenchmarkGraphDRP(
        filepath=bmk.file_path,
        defmodel="graphdrp_default_model.txt",
        # defmodel="graphdrp_model_candle.txt",
        framework="pytorch",
        prog="GraphDRP",
        desc="CANDLE compliant GraphDRP",
    )
    gParameters = candle.finalize_parameters(graphdrp_bmk)
    return gParameters


def main():
    gParameters = initialize_parameters()
    print(gParameters)
    scores = run(gParameters)
    print("Done.")


if __name__ == "__main__":
    main()
