import parsl
from parsl import python_app , bash_app
import subprocess

from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher # USE the MPIExecLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints

from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from time import time
from typing import Sequence, Tuple, Union




#from parsl_apps import preprocess, train, infer
#from IMPROVE.Config.Parsl import Config as Parsl
import csa_params_def as CSA
import improvelib.utils as frm
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import os
from pathlib import Path
import logging
import sys

############
############


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = "auc"

logger = logging.getLogger(f'Start workflow')

class Timer:
  """ Measure time. """
  def __init__(self):
    self.start = time()

  def timer_end(self):
    self.end = time()
    return self.end - self.start

  def display_timer(self, print_fn=print):
    time_diff = self.timer_end()
    if time_diff // 3600 > 0:
        print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
    else:
        print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )

def build_split_fname(source_data_name, split, phase):
    """ Build split file name. If file does not exist continue """
    if split=='all':
        return f"{source_data_name}_{split}.txt"
    return f"{source_data_name}_split_{split}_{phase}.txt"

@python_app  ## May be implemented separately outside this script or does not need parallelization
def preprocess(params, source_data_name, split): # 
    import warnings
    import os
    import subprocess
    import improvelib.utils as frm
    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"

    split_nums=params['split']
    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((params['splits_path']).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((params['splits_path']).glob(f"{source_data_name}_split_{s}_*.txt")))
    files_joined = [str(s) for s in split_files]

    #for split in split_nums:
    print(f"Split id {split} out of {len(split_nums)} splits.")
    # Check that train, val, and test are available. Otherwise, continue to the next split.
    # TODO: check this!
    for phase in ["train", "val", "test"]:
        fname = build_split_fname(source_data_name, split, phase)
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue

    for target_data_name in params['target_datasets']:
        ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"
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
        
        #timer_preprocess = Timer()

        # p1 (none): Preprocess train data
        print("\nPreprocessing")
        train_split_file = f"{source_data_name}_split_{split}_train.txt"
        val_split_file = f"{source_data_name}_split_{split}_val.txt"
        print(f"train_split_file: {train_split_file}")
        print(f"val_split_file:   {val_split_file}")
        print(f"test_split_file:  {test_split_file}")
        print(f"ml_data_outdir:   {params['ml_data_outdir']}")
        if params['use_singularity']:
            preprocess_run = ["singularity", "exec", "--nv",
                params['singularity_image'], "preprocess.sh",
                os.getenv("IMPROVE_DATA_DIR"),
                #str("--x_data_path " + str(params['x_data_path'])),
                #str("--y_data_path " + str(params['y_data_path'])),
                #str("--splits_path " + str(params['splits_path'])),
                #str("--model_specific_outdir " + str(params['model_specific_outdir'])),
                str("--train_split_file " + str(train_split_file)),
                str("--val_split_file " + str(val_split_file)),
                str("--test_split_file " + str(test_split_file)),
                "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
                "--output_dir", str(ml_data_dir),
                str("--y_col_name " + str(params['y_col_name']))
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
        else:
            preprocess_run = ["python",
                params['preprocess_python_script'],
                #"--x_data_path", str(params['x_data_path']),
                #"--y_data_path", str(params['y_data_path']),
                #"--splits_path", str(params['splits_path']),
                #"--model_specific_outdir", str(params['model_specific_outdir']),
                "--train_split_file", str(train_split_file),
                "--val_split_file", str(val_split_file),
                "--test_split_file", str(test_split_file),
                "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
                "--output_dir", str(ml_data_dir),
                "--y_col_name", str(params['y_col_name'])
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
        #timer_preprocess.display_timer(print)
    return {'source_data_name':source_data_name, 'split':split}


@python_app 
def train(params, source_data_name, split): 
    import os
    import warnings
    import subprocess
    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"
    model_outdir = params['model_outdir']/f"{source_data_name}"/f"split_{split}"
    #frm.create_outdir(outdir=model_outdir)
    #for target_data_name in params['target_datasets']:
    ml_data_outdir = params['ml_data_dir']/f"{source_data_name}-{params['target_datasets'][0]}"/f"split_{split}"  #### We cannot have target data name here ??????
    if model_outdir.exists() is False:
        os.makedirs(os.path.join(model_outdir, 'ckpts'), exist_ok=True) # For storing checkpoints
        train_ml_data_dir = ml_data_outdir
        val_ml_data_dir = ml_data_outdir
        #timer_train = Timer()
        print("\nTrain")
        print(f"train_ml_data_dir: {train_ml_data_dir}")
        print(f"val_ml_data_dir:   {val_ml_data_dir}")
        print(f"model_outdir:      {model_outdir}")
        if params['use_singularity']:
            train_run = ["singularity", "exec", "--nv",
                    params['singularity_image'], "train.sh", '$CUDA_VISIBLE_DEVICES',
                    os.getenv("IMPROVE_DATA_DIR"),
                    str("--train_ml_data_dir " + str(train_ml_data_dir)),
                    str("--val_ml_data_dir " + str(val_ml_data_dir)),
                    str("--ml_data_outdir " + str(ml_data_outdir)),
                    str("--model_specific_outdir " + str(params['model_specific_outdir'])),
                    str("--model_outdir " + str(model_outdir)),
                    str("--epochs " + str(params['epochs'])),
                    str("--y_col_name " + params['y_col_name']),
                    str("--ckpt_directory " + os.path.join(model_outdir, 'ckpts'))
            ]
            result = subprocess.run(train_run, capture_output=True,
                                    text=True, check=True)
        else:
            train_run = ["CUDA_VISIBLE_DEVICES=", "$CUDA_VISIBLE_DEVICES", "python",
                "train.py",
                "--train_ml_data_dir", str(train_ml_data_dir),
                "--val_ml_data_dir", str(val_ml_data_dir),
                "--ml_data_outdir", str(ml_data_outdir),
                "--model_specific_outdir", str(params['model_specific_outdir']),
 
                "--model_outdir", str(model_outdir),
                "--epochs", str(params['epochs']),
                "--y_col_name", params['y_col_name'],
                "--ckpt_directory", os.path.join(model_outdir, 'ckpts')
            ]
            result = subprocess.run(train_run, capture_output=True,
                                    text=True, check=True)
    return {'source_data_name':source_data_name, 'split':split}

@python_app  
def infer(params, source_data_name, target_data_name, split): # 
    import os
    import warnings
    import subprocess
    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"
    #for split in params['split']:
    ml_data_outdir = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    model_outdir = params['model_outdir']/f"{source_data_name}"/f"split_{split}"
    test_ml_data_dir = ml_data_outdir
    infer_outdir = params['infer_outdir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    #timer_infer = Timer()

    print("\nInfer")
    print(f"test_ml_data_dir: {test_ml_data_dir}")
    print(f"infer_outdir:     {infer_outdir}")
    if params['use_singularity']:
        infer_run = ["singularity", "exec", "--nv",
                params['singularity_image'], "infer.sh",'$CUDA_VISIBLE_DEVICES',
                os.getenv("IMPROVE_DATA_DIR"),
                str("--test_ml_data_dir "+ str(test_ml_data_dir)),
                str("--model_dir " + str(model_outdir)),
                str("--infer_outdir " + str(infer_outdir)),
                str("--model_specific_outdir " + str(params['model_specific_outdir'])),
                str("--y_col_name " + params['y_col_name']),
                str("--model_outdir " + str(model_outdir)),
                str("--ml_data_outdir " + str(ml_data_outdir))
        ]
        result = subprocess.run(infer_run, capture_output=True,
                                text=True, check=True)
    else:
        infer_run = ["python",
                "infer.py",
                "--test_ml_data_dir", str(test_ml_data_dir),
                "--model_dir", str(model_outdir),
                "--infer_outdir", str(infer_outdir),
                "--y_col_name", params['y_col_name'],
                "--model_outdir", str(model_outdir),
                "--model_specific_outdir", str(params['model_specific_outdir']),
                "--ml_data_outdir", str(ml_data_outdir)
        ]
        result = subprocess.run(infer_run, capture_output=True,
                                text=True, check=True)
    #timer_infer.display_timer(print)
    return True

############
############

def build_split_fname(source_data_name, split, phase):
    """ Build split file name. If file does not exist continue """
    if split=='all':
        return f"{source_data_name}_{split}.txt"
    return f"{source_data_name}_split_{split}_{phase}.txt"


additional_definitions = CSA.additional_definitions
filepath = Path(__file__).resolve().parent


## Should we combine csa config and parsl config and use just one initialize_parameter??
cfg = DRPPreprocessConfig() # TODO submit github issue; too many logs printed; is it necessary?
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.ini",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=additional_definitions,
    required=None
)

print(params)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']


logger = logging.getLogger(f"{params['model_name']}")

params = frm.build_paths(params)  # paths to raw data
MAIN_CSA_OUTDIR = Path(f"./0_{y_col_name}_improvelib_small")
params['ml_data_dir'] = MAIN_CSA_OUTDIR / 'ml_data'  ### May be add to frm.build_paths()??
params['model_outdir'] = MAIN_CSA_OUTDIR / 'models'
params['infer_outdir'] = MAIN_CSA_OUTDIR / 'infer'
#params['model_specific_outdir'] = MAIN_CSA_OUTDIR/params['model_specific_outdir']
#Model scripts
params['preprocess_python_script'] = f"{params['model_name']}_preprocess_improve.py"
params['train_python_script'] = f"{params['model_name']}_train_improve.py"
params['infer_python_script'] = f"{params['model_name']}_infer_improve.py"

### Initialize params reads as strings. So adding non str params here:
params['use_singularity'] = False
params['model_specific_data'] = False
params['source_datasets'] = ['CCLE', 'gCSI']
params['target_datasets'] = ["CCLE", "gCSI"]
params['split'] = ['0']
params['only_cross_study'] = False
params['epochs'] = 1

##TODO: Also download benchmark data here

## Download Author specific data
if params['model_specific_data']:
    auth_data_download = ["bash",
        "model_specific_data_download.sh",
        str(params['model_specific_data_url']),
        str(params['model_specific_outdir'])
    ]
    result = subprocess.run(auth_data_download, capture_output=True,
                            text=True, check=True)


##### CONFIG FOR LAMBDA ######
# Adjust your user-specific options here:
run_dir="~/tmp"
print(parsl.__version__)

available_accelerators: Union[int, Sequence[str]] = 2
worker_port_range: Tuple[int, int] = (10000, 20000)
retries: int = 1

config_lambda = Config(
    retries=retries,
    executors=[
        HighThroughputExecutor(
            address="127.0.0.1",
            label="htex_Local",
            cpu_affinity="block",
            #max_workers_per_node=2,
            worker_debug=True,
            available_accelerators=available_accelerators,
            worker_port_range=worker_port_range,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
)
""" user_opts = {
    "worker_init":      f"source ~/.venv/parsl/bin/activate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE",
    "queue":            "R1819593",
    "walltime":         "1:00:00",
    "nodes_per_block":  10, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
} """

user_opts = {
    "worker_init":      f". ~/.bashrc ; conda activate parsl; export PYTHONPATH=$PYTHONPATH:/IMPROVE; export IMPROVE_DATA_DIR=./improve_dir; module use /soft/spack/gcc/0.6.1/install/modulefiles/Core; module load apptainer; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE_Aim1",
    "queue":            "debug-scaling",
    "walltime":         "1:00:00",
    "nodes_per_block":  3,# think of a block as one job on polaris, so to run on the main queues, set this >= 10
}


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

##################### START PARSL PARALLEL EXECUTION #####################
train_futures=[]

#parsl.load(local_config)
parsl.load(config_lambda)
for source_data_name in params['source_datasets']:
    for split in params['split']:
        for target_data_name in params['target_datasets']:
            preprocess_futures = preprocess(params, source_data_name, split)  ## MODIFY TO INCLUDE SPLITS IN PARALLEL?
            #train_futures.append(train(params, preprocess_futures.result()['source_data_name'], preprocess_futures.result()['split']))
            #train_future = train(params, preprocess_futures.result()['source_data_name'], preprocess_futures.result()['split'])
            #infer_futures = infer(params, train_future.result()['source_data_name'], target_data_name, train_future.result()['split'])


#for target_data_name in params['target_datasets']:
#    infer_futures = [infer(params, tf.result()['source_data_name'], target_data_name, tf.result()['split']) for tf in train_futures]

parsl.clear()

### module load apptainer