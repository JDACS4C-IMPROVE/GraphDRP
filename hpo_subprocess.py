"""
Before running this script, first need to preprocess the data.
This can be done by running preprocess_example.sh

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $IMPROVE_DATA_DIR and $PYTHONPATH are set:
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/path/to/IMPROVE_lib

mpirun -np 2 python hpo_subprocess.py

TODO: how to distribute HPO to mulitple GPUs?
"""
# import copy
import json
import subprocess
import os
import pandas as pd

from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# breakpoint()
num_gpus_per_node = 4
gpu_id0 = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus_per_node + gpu_id0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# ---------------------
# Hyperparameters
# ---------------------
problem = HpProblem()

problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
problem.add_hyperparameter((1e-6, 1e-2, "log-uniform"),
                           "learning_rate", default_value=0.001)
# problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
# problem.add_hyperparameter([True, False], "early_stopping", default_value=False)

# ---------------------
# Some IMPROVE settings
# ---------------------
# source = "CCLE"
source = "CTRPv2"
# source = "gCSI"
# source = "GDSCv1"
# source = "GDSCv2"
split = 0
train_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
val_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
model_outdir = f"{source}_dh_hpo_improve/{source}/split_{split}"
log_dir = f"{source}_dh_hpo_logs/"
subprocess_bashscript = "subprocess_train.sh"


@profile
def run(job, optuna_trial=None):

    # config = copy.deepcopy(job.parameters)
    # params = {
    #     "epochs": DEEPHYPER_BENCHMARK_MAX_EPOCHS,
    #     "timeout": DEEPHYPER_BENCHMARK_TIMEOUT,
    #     "verbose": False,
    # }
    # if len(config) > 0:
    #     remap_hyperparameters(config)
    #     params.update(config)

    model_outdir_job_id = model_outdir + f"/{job.id}"

    # val_scores = main_train_grapdrp([
    #     "--train_ml_data_dir", str(train_ml_data_dir),
    #     "--val_ml_data_dir", str(val_ml_data_dir),
    #     "--model_outdir", str(model_outdir_job_id),
    # ])
    
    subprocess_res = subprocess.run(
        [
            "bash", subprocess_bashscript,
             str(train_ml_data_dir),
             str(val_ml_data_dir),
             str(model_outdir_job_id)
        ], 
        capture_output=True, text=True, check=True
    )
    
    # print(subprocess_res.stdout)
    # print(subprocess_res.stderr)

    # Load val_scores and get val_loss
    # f = open(model_outdir + "/val_scores.json")
    f = open(model_outdir_job_id + "/val_scores.json")
    val_scores = json.load(f)
    objective = -val_scores["val_loss"]
    # print("objective:", objective)

    # Checkpoint the model weights
    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    import time
    t0 = time.time()
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)

            search = CBO(
                problem,
                evaluator,
                log_dir=log_dir,
                verbose=1,
            )

            # max_evals = 2
            # max_evals = 4
            # max_evals = 10
            # max_evals = 20
            max_evals = 50
            # max_evals = 100
            results = search.search(max_evals=max_evals)
            results = results.sort_values("m:val_loss", ascending=True)
            results.to_csv(model_outdir + "/hpo_results.csv", index=False)

    print(f"Finished deephyper HPO (rank {rank}) in {(time.time() - t0) / 60} mins.")
