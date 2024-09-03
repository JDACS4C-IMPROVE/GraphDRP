""" Run this script to generate dataframes for plotting for Uno. Then run
ovarian_postprocess.ipynb
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = Path(__file__).parent  # py
# filepath = Path(os.path.abspath(''))  # ipynb
print(filepath)

# Models dir
models_dir = filepath / "pred_models"

canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"
y_col_name = "auc_pred"

ov_drug_info = pd.read_csv(filepath / "ovarian_data/raw_data/y_data/Drugs_For_OV_Proposal_Analysis.txt", sep="\t")

# -----------------------------------------
# IMPROVE models
# model_name  = 'DeepTTC'
# model_name  = 'GraphDRP'
# model_name  = 'HIDRA'
# model_name  = 'IGTD'
# model_name  = 'PaccMann_MCA'
# infer_dir = models_dir / model_name / "out_infer"
preds_fname = "test_y_data_predicted.csv"   # preds file

# UNO
model_name  = 'UNO'
infer_dir = models_dir / model_name / "out_infer"  # infer dir
preds_fname = "predicted.all.tsv"  # preds file

source = "all"
# source = "CTRPv2"
# # source = "GDSCv2"
target = "PDMR"
# -----------------------------------------


def agg_pred_scores(df: pd.DataFrame):
    """ Take the aggregated file with raw predictions from all splits and
    aggregate via mean and std across splits. """
    group_by_cols = [canc_col_name, drug_col_name]
    ff = df.groupby(group_by_cols).agg(
        pred_mean=(y_col_name, 'mean'),
        pred_std=(y_col_name, 'std'))
    ff = ff.reset_index().sort_values([drug_col_name, canc_col_name, 'pred_mean']).reset_index(drop=True)
    return ff


def check_dups(df):
    tt = df[df.duplicated(keep=False)]
    assert tt.shape[0] == 0, "found duplicates"


preds_fpath = infer_dir / preds_fname
df = pd.read_csv(preds_fpath, sep="\t")
df["model"] = model_name
df = df.rename(columns={'Sample': canc_col_name,
                        'Drug1': drug_col_name,
                        'PredictedAUC': y_col_name})
df[y_col_name] = pd.to_numeric(df[y_col_name], errors='coerce')
df = df.dropna(subset=[y_col_name])
df = df.reset_index(drop=True)
df = df.drop_duplicates()

agg_df = agg_pred_scores(df)
# agg_df["model"] = model_name
check_dups(df)

# preds_fpath = infer_dir / preds_fname
# agg_df = pd.read_csv(preds_fpath, sep="\t")
# print(agg_df.shape)
# agg_df = agg_df.drop_duplicates()
# print(agg_df.shape)
# # check_dups(agg_df)
# agg_df = agg_df.rename(columns={'Sample': canc_col_name,
#                                 'Drug1': drug_col_name,
#                                 'PredictedAUC': y_col_name})

ov_drug_info = ov_drug_info[[drug_col_name, 'drug_name']].sort_values(drug_col_name)
agg_df = agg_df.merge(ov_drug_info, on=drug_col_name, how="inner").drop_duplicates()
check_dups(agg_df)

plots_outdir = filepath / 'plots_outdir'
os.makedirs(plots_outdir, exist_ok=True)
df.to_csv(plots_outdir / f"all_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)
agg_df.to_csv(plots_outdir / f"agg_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)

print("Finished")
