""" Run this script to generate dataframes for plotting. Then run ovarian_postprocess.ipynb """
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = Path(__file__).parent  # py
# filepath = Path(os.path.abspath(''))  # ipynb
print(filepath)

# infer_dir = filepath / "out_infer"
models_dir = filepath / "pred_models"
model_name  = 'DeepTTC'
# model_name  = 'GraphDRP'
# model_name  = 'HIDRA'
# model_name  = 'IGTD'
# model_name  = 'PaccMann_MCA'
infer_dir = models_dir / model_name / "out_infer"

preds_fname = "test_y_data_predicted.csv"

canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"

source = "CTRPv2"
# source = "GDSCv2"
target = "PDMR"

ov_drug_info = pd.read_csv(filepath / "ovarian_data/raw_data/y_data/Drugs_For_OV_Proposal_Analysis.txt", sep="\t")


def agg_pred_files(res_dir: Path):
    dfs = []
    split_dirs = list(res_dir.glob("split_*"))
    for sdir in split_dirs:
        preds_fpath = sdir / preds_fname
        if preds_fpath.exists():
            split = str(sdir).split("split_")[1]
            df = pd.read_csv(preds_fpath)
            df["split"] = int(split)
            dfs.append(df)
    rr = pd.concat(dfs, axis=0)#.sort_values(["split"]).reset_index(drop=True)
    rr = rr.sort_values(["split", canc_col_name, drug_col_name]).reset_index(drop=True)
    return rr


def agg_pred_scores(df: pd.DataFrame):
    group_by_cols = ['improve_sample_id', 'improve_chem_id']
    ff = df.groupby(group_by_cols).agg(
        pred_mean=('auc_pred', 'mean'),
        pred_std=('auc_pred', 'std'))
    ff = ff.reset_index().sort_values(['improve_chem_id', 'improve_sample_id', 'pred_mean']).reset_index(drop=True)
    # ff = ff.reset_index().sort_values(['pred_mean', 'improve_chem_id', 'improve_sample_id']).reset_index(drop=True)
    return ff


def check_dups(df):
    tt = df[df.duplicated(keep=False)]
    assert tt.shape[0] == 0, "found duplicates"


res_dir = infer_dir / f"{source}-{target}"
df = agg_pred_files(res_dir)
df["model"] = model_name
# tt = df[df.duplicated(keep=False)]
check_dups(df)
agg_df = agg_pred_scores(df)
agg_df["model"] = model_name
check_dups(agg_df)

ov_drug_info = ov_drug_info[[drug_col_name, 'drug_name']].sort_values(drug_col_name)
# print(kk)
agg_df = agg_df.merge(ov_drug_info, on=drug_col_name, how="inner").drop_duplicates()
check_dups(agg_df)

plots_outdir = filepath / 'plots_outdir'
os.makedirs(plots_outdir, exist_ok=True)
df.to_csv(plots_outdir / f"all_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)
agg_df.to_csv(plots_outdir / f"agg_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)

print("Finished")