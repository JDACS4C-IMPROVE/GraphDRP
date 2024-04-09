"""
python csa_postproc.py --res_dir auc_old --model_name GraphDRP --y_col_name auc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# IMPROVE/CANDLE imports
# from improve import framework as frm
# from improve.csa import cross_study_postprocess
from improve.evaluation_schemes.cross_study_analysis import cross_study_postprocess

# Imports from preprocess script
# from graphdrp_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', required=False, default='auc', type=str,
                    help='Dir containing the results.')
parser.add_argument('--model_name', required=False, default='GraphDRP', type=str,
                    help='Name of the model.')
parser.add_argument('--y_col_name', required=False, default='auc', type=str,
                    help='Y col name.')
args = parser.parse_args()

res_dir = args.res_dir
model_name = args.model_name
y_col_name = args.y_col_name

# breakpoint()
res_dir_path = filepath/res_dir
outdir = res_dir_path/f"../res.csa.{model_name}.{res_dir}"
scores = cross_study_postprocess(res_dir_path,
                                 model_name,
                                 y_col_name,
                                 outdir=outdir)
# breakpoint()
print("\nFinished cross-study post-processing.")
