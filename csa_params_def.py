import os
from pathlib import Path
#import model_specific_config - To import model specific parameters


##TODO Replace CANDLE initialize_parameter()

fdir = Path(__file__).resolve().parent
required = None
additional_definitions = [
    {"name": "input_dir",
     "type": str,
     "default": 'input',
     "help": "Input directory containing the raw data"
    },
    {"name": "main_csa_outdir",
     "type": str,
     "default": 'model',
     "help": "Parent output directory containing the preprocessed data, trained models and inference results"
    },
    {"name": "source_datasets",
     "nargs" : "+",
     "type": str,
     "default": ['CCLE'],
     "help": "source_datasets for cross study analysis"
    },
    {"name": "target_datasets",
     "nargs" : "+",
     "type": str,
     "default": ["CCLE", "gCSI"],
     "help": "target_datasets for cross study analysis"
    },
    {"name": "split",
     "nargs" : "+",
     "type": str,
     "default": ['0'],
     "help": "Split number for preprocessing"
    },
    {"name": "only_cross_study",
     "type": bool,
     "default": False,
     "help": "If only cross study analysis is needed"
    },
    {"name": "log_level",
     "type": str,
     "default": os.getenv("IMPROVE_LOG_LEVEL", "WARNING"),
     "help": "Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET"
    },
    {"name": "model_name",
     "type": str,
     "default": 'graphdrp', ## Change the default to LGBM??
     "help": "Name of the deep learning model"
    },
    {"name": "model_specific_data",
     "type": bool,
     "default": False, ## Change the default to LGBM??
     "help": "Use model specific data?"
    },
    {"name": "epochs",
     "type": int,
     "default": 10,
     "help": "Number of epochs"
    },
    {"name": "use_singularity",
     "type": bool,
     "default": True,
     "help": "Do you want to use singularity image for running the model?"
    },
    {"name": "singularity_image",
     "type": str,
     "default": '',
     "help": "Singularity image file of the model"
    }
    ]