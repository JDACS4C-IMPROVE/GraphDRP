import parsl
from parsl import python_app
import subprocess
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from time import time
from typing import Sequence, Tuple, Union
from pathlib import Path
import logging
import sys

import csa_params_def as CSA
import improvelib.utils as frm
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig



##### CONFIG FOR LAMBDA ######
available_accelerators: Union[int, Sequence[str]] = 8
worker_port_range: Tuple[int, int] = (10000, 20000)
retries: int = 1

config_lambda = Config(
    retries=retries,
    executors=[
        HighThroughputExecutor(
            address='127.0.0.1',
            label="htex",
            cpu_affinity="block",
            #max_workers_per_node=2, ## IS NOT SUPPORTED IN  Parsl version: 2023.06.19. CHECK HOW TO USE THIS???
            worker_debug=True,
            available_accelerators=4,  ## CHANGE THIS AS REQUIRED BY THE MACHINE
            worker_port_range=worker_port_range,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy='simple',
)

parsl.clear()
parsl.load(config_lambda)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
logger = logging.getLogger(f'Start workflow')

##############################################################################
################################ PARSL APPS ##################################
##############################################################################

@python_app  
def preprocess(inputs=[]): # 
    import warnings
    import os
    import subprocess
    import improvelib.utils as frm
    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"
    params=inputs[0]
    source_data_name=inputs[1]
    split=inputs[2]

    split_nums=params['split']
    # Get the split file paths
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((params['splits_path']).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
    else:
        split_files = []
        for s in split_nums:
            split_files.extend(list((params['splits_path']).glob(f"{source_data_name}_split_{s}_*.txt")))
    files_joined = [str(s) for s in split_files]

    print(f"Split id {split} out of {len(split_nums)} splits.")
    # Check that train, val, and test are available. Otherwise, continue to the next split.
    for phase in ["train", "val", "test"]:
        fname = build_split_fname(source_data_name, split, phase)
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue

    for target_data_name in params['target_datasets']:
        ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/ \
                f"split_{split}"
        if ml_data_dir.exists() is True:
            continue
        if params['only_cross_study'] and (source_data_name == target_data_name):
            continue # only cross-study
        print(f"\nSource data: {source_data_name}")
        print(f"Target data: {target_data_name}")

        params['ml_data_outdir'] = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
        frm.create_outdir(outdir=params["ml_data_outdir"])
        if source_data_name == target_data_name:
            # If source and target are the same, then infer on the test split
            test_split_file = f"{source_data_name}_split_{split}_test.txt"
        else:
            # If source and target are different, then infer on the entire target dataset
            test_split_file = f"{target_data_name}_all.txt"

        # Preprocess  data
        print("\nPreprocessing")
        train_split_file = f"{source_data_name}_split_{split}_train.txt"
        val_split_file = f"{source_data_name}_split_{split}_val.txt"
        print(f"train_split_file: {train_split_file}")
        print(f"val_split_file:   {val_split_file}")
        print(f"test_split_file:  {test_split_file}")
        print(f"ml_data_outdir:   {params['ml_data_outdir']}")
        if params['use_singularity']:
            print('Functionality using singularity is work in progress. Please use the Python version to call preprocess, set use_singularity=False')

        else:
            preprocess_run = ["python",
                params['preprocess_python_script'],
                #"--model_specific_outdir", str(params['model_specific_outdir']),
                "--train_split_file", str(train_split_file),
                "--val_split_file", str(val_split_file),
                "--test_split_file", str(test_split_file),
                "--input_dir", params['input_dir'], 
                "--output_dir", str(ml_data_dir),
                "--y_col_name", str(params['y_col_name'])
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
    return {'source_data_name':source_data_name, 'split':split}


@python_app 
def train(params, source_data_name, split): 
    import os
    import warnings
    import subprocess
    model_dir = params['model_outdir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{params['target_datasets'][0]}"/ \
                f"split_{split}"
    if model_dir.exists() is False:
        print("\nTrain")
        print(f"ml_data_dir: {ml_data_dir}")
        print(f"model_dir:   {model_dir}")
        if params['use_singularity']:
            print('Functionality using singularity is work in progress. Please use the Python version to call train, set use_singularity=False')
        else:
            train_run = ["python", 
                         params['train_python_script'],
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(params['epochs']),  # DL-specific
                        "--y_col_name", str(params['y_col_name'])
                    ]
            result = subprocess.run(train_run, capture_output=True,
                                    text=True, check=True)
    return {'source_data_name':source_data_name, 'split':split}

@python_app  
def infer(params, source_data_name, target_data_name, split): # 
    import os
    import warnings
    import subprocess
    model_dir = params['model_outdir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{params['target_datasets'][0]}"/ \
                f"split_{split}"
    infer_dir = params['infer_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    if params['use_singularity']:
        print('Functionality using singularity is work in progress. Please use the Python version to call infer, set use_singularity=False')

    else:
        print("\nInfer")
        infer_run = ["python", params['infer_python_script'],
                "--input_data_dir", str(ml_data_dir),
                "--input_model_dir", str(model_dir),
                "--output_dir", str(infer_dir),
                "--y_col_name", str(params['y_col_name'])
        ]
        result = subprocess.run(infer_run, capture_output=True,
                                text=True, check=True)
    return True

###############################
####### CSA PARAMETERS ########
###############################

additional_definitions = CSA.additional_definitions
filepath = Path(__file__).resolve().parent
cfg = DRPPreprocessConfig() # TODO submit github issue; too many logs printed; is it necessary?
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.ini",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=additional_definitions,
    required=None
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']
logger = logging.getLogger(f"{params['model_name']}")
params = frm.build_paths(params)  # paths to raw data

#Output directories for preprocess, train and infer
params['ml_data_dir'] = Path(params['output_dir']) / 'ml_data' 
params['model_outdir'] = Path(params['output_dir']) / 'models'
params['infer_dir'] = Path(params['output_dir']) / 'infer'
#params['model_specific_outdir'] = Path(params['output_dir']) / params['model_specific_outdir'] # CHANGE THIS

#Model scripts
params['preprocess_python_script'] = f"{params['model_name']}_preprocess_improve.py"
params['train_python_script'] = f"{params['model_name']}_train_improve.py"
params['infer_python_script'] = f"{params['model_name']}_infer_improve.py"

## Download Author specific data ----> MOVE IT TO setup_improve.sh
if params['model_specific_data']:
    auth_data_download = ["bash",
        "model_specific_data_download.sh",
        str(params['model_specific_data_url']),
        str(params['model_specific_outdir'])
    ]
    result = subprocess.run(auth_data_download, capture_output=True,
                            text=True, check=True)

##########################################################################
##################### START PARSL PARALLEL EXECUTION #####################
##########################################################################

##Preprocess execution with Parsl
preprocess_futures=[]
for source_data_name in params['source_datasets']:
    for split in params['split']:
            preprocess_futures.append(preprocess(inputs=[params, source_data_name, split])) 

##Train execution with Parsl
train_futures=[]
for future_p in preprocess_futures:
    train_futures.append(train(params, future_p.result()['source_data_name'], future_p.result()['split']))

##Infer execution with Parsl
infer_futures =[]
for future_t in train_futures:
    for target_data_name in params['target_datasets']:
        infer_futures.append(infer(params, future_t.result()['source_data_name'], target_data_name, future_t.result()['split']))

for future_i in infer_futures:
    print(future_i.result())

## TODO: PARSL CONFIG FOR POLARIS
""" user_opts = {
    "worker_init":      f"source ~/.venv/parsl/bin/activate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE",
    "queue":            "R1819593",
    "walltime":         "1:00:00",
    "nodes_per_block":  10, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
} 

user_opts = {
    "worker_init":      f". ~/.bashrc ; conda activate parsl; export PYTHONPATH=$PYTHONPATH:/IMPROVE; export IMPROVE_DATA_DIR=./improve_dir; module use /soft/spack/gcc/0.6.1/install/modulefiles/Core; module load apptainer; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE_Aim1",
    "queue":            "debug-scaling",
    "walltime":         "1:00:00",
    "nodes_per_block":  3,# think of a block as one job on polaris, so to run on the main queues, set this >= 10
}
"""

""" 
####### CONFIG FOR POLARIS ######

config_polaris = Config(
            retries=1,  # Allows restarts if jobs are killed by the end of a job
            executors=[
                HighThroughputExecutor(
                    label="htex",
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    max_workers=64,
                    available_accelerators=4,  # Ensures one worker per accelerator
                    address=address_by_interface("bond0"),
                    cpu_affinity="block-reverse",
                    prefetch_capacity=0,  # Increase if you have many more tasks than workers
                    start_method="spawn",
                    provider=PBSProProvider(  # type: ignore[no-untyped-call]
                        launcher=MpiExecLauncher(  # Updates to the mpiexec command
                            bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                        ),
                        account="IMPROVE_Aim1",
                        queue="debug-scaling",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=user_opts['scheduler_options'],
                        worker_init=user_opts['worker_init'],
                        nodes_per_block=10,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel jobs
                        cpus_per_node=64,
                        walltime="1:00:00",
                    ),
                ),
            ],
            run_dir=str(run_dir),
            strategy='simple',
            app_cache=True,
        )  """

""" config_polaris = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                available_accelerators=4, # if this is set, it will override other settings for max_workers if set
                max_workers_per_node=4, # Set as many workers as there are GPUs because we want one worker to use 1 GPU
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                worker_debug=True,
                # start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1, # Can increase more to have more parallel jobs
                    # cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"]
                ),
            ),
        ],
        retries=2,
        app_cache=True,
) """