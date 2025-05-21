"""
Model-specific params (Model: GraphDRP)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = []


train_params = [
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
]


infer_params = [   
    {"name": "model_arch",
     "type": str,
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "help": "Model architecture to run."
    }, 
    {"name": "cuda_name",
     "type": str,
     # "action": "store",
     "default": "cuda:0",
     "help": "Cuda device (e.g.: cuda:0, cuda:1)."
    },
]
