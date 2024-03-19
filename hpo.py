"""
Before running hpo.py, first need to preprocess the data.
This can be done by running preprocess_example.sh

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $IMPROVE_DATA_DIR and $PYTHONPATH are set:
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/path/to/IMPROVE_lib
"""
import copy

from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO

# Import the function that returns the objective to be maximized
# from .graphdrp_train_improve_dh import run as run_train_grapdrp
from .graphdrp_train_improve_dh import main as main_train_grapdrp


# ---------------------
# Model hyperparameters
# ---------------------
problem = HpProblem()

# problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
problem.add_hyperparameter((1e-6, 1e-2, "log-uniform"), "learning_rate", default_value=0.001)

# problem.add_hyperparameter([True, False], "early_stopping", default_value=False)
# problem.add_hyperparameter((5, 20), "early_stopping_patience", default_value=5)

# ---------------------
# Some IMPROVE settings
# ---------------------
source = "CCLE"
split = 0
train_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
val_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
model_outdir = f"out_models_hpo/{source}/split_{split}"
log_dir = "hpo_logs/"


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

    model_outdir_job_id = model_outdir + f"/{job_id}"

    val_scores = main_train_grapdrp([
        "--train_ml_data_dir", str(train_ml_data_dir),
        "--val_ml_data_dir", str(val_ml_data_dir),
        "--model_outdir", str(model_outdir_job_id),
    ])

    objective = -val_scores["val_loss"]

    # Checkpoint the model weights
    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    # return {"objective": objective, "metadata": metadata}
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
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

            results = search.search(max_evals=100)
