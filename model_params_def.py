"""
Model-specific params (Model: GraphDRP)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = [
    {"name": "use_lincs",
     "type": str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
    },
]


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
