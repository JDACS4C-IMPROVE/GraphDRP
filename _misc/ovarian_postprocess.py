""" Run this script to generate dataframes for plotting. Then run
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
model_name  = 'PaccMann_MCA'
infer_dir = models_dir / model_name / "out_infer"  # infer dir
preds_fname = "test_y_data_predicted.csv"   # preds file

source = "CTRPv2"
# source = "GDSCv2"
target = "PDMR"
# -----------------------------------------


def agg_pred_files(res_dir: Path):
    """ Load files with raw predictions from different data splits and combine
    into a single dataframe. """
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


res_dir = infer_dir / f"{source}-{target}"
df = agg_pred_files(res_dir)
df["model"] = model_name
check_dups(df)

agg_df = agg_pred_scores(df)
agg_df["model"] = model_name
check_dups(agg_df)

ov_drug_info = ov_drug_info[[drug_col_name, 'drug_name']].sort_values(drug_col_name)
agg_df = agg_df.merge(ov_drug_info, on=drug_col_name, how="inner").drop_duplicates()
check_dups(agg_df)

plots_outdir = filepath / 'plots_outdir'
os.makedirs(plots_outdir, exist_ok=True)
df.to_csv(plots_outdir / f"all_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)
agg_df.to_csv(plots_outdir / f"agg_preds_{model_name}_{source}_{target}.tsv", sep="\t", index=False)

print("Finished")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# # df.DataFrame(data)
# pdo_name = '655913~031-T'

# # Calculate average pred_mean for each drug and sort
# drug_avg = df.groupby('drug_name')['pred_mean'].mean().sort_values(ascending=False)
# drug_order = drug_avg.index

# # Set up the plot
# fig, ax = plt.subplots(figsize=(20, 12))

# # Set the width of each bar and the positions of the bars
# width = 0.15
# x = range(len(drug_order))

# # Create the grouped bar plot
# for i, model in enumerate(['DeepTTC', 'GraphDRP', 'HIDRA', 'IGTD', 'PaccMann_MCA']):
#     data = df[df['model'] == model].set_index('drug_name').loc[drug_order]
#     bars = ax.bar([xi + i*width for xi in x], data['pred_mean'], width, label=model)
    
#     # Add text labels
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)


# # Customize the plot
# ax.set_ylabel('Prediction Mean', fontsize=12)
# ax.set_xlabel('Drug', fontsize=12)
# ax.set_title(f'{}: Drug Prediction Comparison Across Models', fontsize=16)
# ax.set_xticks([xi + 2*width for xi in x])
# ax.set_xticklabels(drug_order, rotation=45, ha='right', fontsize=10)
# ax.legend(fontsize=10)

# # Add grid
# ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# # Set y-axis limits
# ax.set_ylim(0, 1.1)

# # Adjust the layout and display the plot
# plt.tight_layout()
# plt.show()