"""
Calculate summary values for each model.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, sem

fdir = Path(__file__).resolve().parent


# ====================
# Summary table
# ====================
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', required=False, default='LGBM', type=str,
#                     help='Name of the model.')
# args = parser.parse_args()
# model_name = args.model_name
# datadir = fdir/f"results.{model_name}"
# outdir = fdir/f"scores.{model_name}"
# scores_names = ["mae", "r2", "pcc", "scc"]

# import ipdb; ipdb.set_trace(context=5)
# # Data source study
# sc_df = []
# cols = ["Metric", "Diag", "Off-diag"]
# for i, sc_name in enumerate(scores_names):
#     print("\nMetric:", sc_name)
#     sc_item = {}
#     sc_item["Metric"] = sc_name.upper()
#     mean_df = pd.read_csv(outdir/f"{sc_name}_mean_table.csv")
#     n = mean_df.shape[0]

#     # Diag
#     vv = np.diag(mean_df.iloc[:, 1:].values)
#     # s1 = sum(vv)
#     sc_item["Diag"] = np.round(sum(vv)/n, 3)
#     sc_item["Diag_std"] = np.round(np.std(vv), 3)

#     # Off-diag
#     vv = mean_df.iloc[:, 1:].values
#     np.fill_diagonal(vv, 0)
#     sc_item["Off-diag"] = np.round(sum(np.ravel(vv)) / (n*n - n), 3)
#     sc_item["Off-diag_std"] = np.round(np.std(np.ravel(vv)), 3)

#     for ii, dname in enumerate(mean_df.iloc[:, 0].values):
#         dname = dname.upper()
#         sc_item[dname] = np.round(sum(vv[ii, :] / (n - 1)), 3)
#         if i == 0:
#             cols.append(dname)

#     sc_df.append(sc_item)

# sc_df = pd.DataFrame(sc_df, columns=cols)
# sc_df.to_csv(outdir/f"{summary}_table.csv", index=True)
# print(sc_df)


# ====================
# Summary table
# ====================
import ipdb; ipdb.set_trace(context=5)
outdir = fdir
model_names = ["DeepTTC", "GraphDRP_03", "HIDRA", "IGTD"]
# Data source study
dt = []
diag_dfs = []
diag_dfs_std = []
off_diag_dfs = []
off_diag_dfs_std = []

for i, model_name in enumerate(model_names):
    print("Model:", model_name)

    df = pd.read_csv(fdir/f"scores.{model_name}"/"summary_table.csv")
    diag_dfs.append(df[["Metric", "Diag"]].rename(columns={"Diag": model_name}))
    off_diag_dfs.append(df[["Metric", "Off-diag"]].rename(columns={"Off-diag": model_name}))
    diag_dfs_std.append(df[["Metric", "Diag_std"]].rename(columns={"Diag_std": model_name}))
    off_diag_dfs_std.append(df[["Metric", "Off-diag_std"]].rename(columns={"Off-diag_std": model_name}))

    df = df.drop(columns=["Diag", "Off-diag", "Diag_std", "Off-diag_std"])
    df.insert(loc=1, column="Model", value=model_name)
    dt.append(df)

# import ipdb; ipdb.set_trace(context=5)
dg = pd.concat(diag_dfs, axis=1)
dg = dg.T.drop_duplicates().T
print(dg)
dg_std = pd.concat(diag_dfs_std, axis=1)
dg_std = dg_std.T.drop_duplicates().T
print(dg_std)
odg = pd.concat(off_diag_dfs, axis=1)
odg = odg.T.drop_duplicates().T
print(odg)
odg_std = pd.concat(off_diag_dfs_std, axis=1)
odg_std = odg_std.T.drop_duplicates().T
print(odg_std)
dt = pd.concat(dt, axis=0).sort_values(["Metric", "Model"])
print(dt)

dg.to_csv(outdir/f"diag_table.csv", index=False)
dg_std.to_csv(outdir/f"diag_std_table.csv", index=False)
odg.to_csv(outdir/f"off_diag_table.csv", index=False)
odg_std.to_csv(outdir/f"off_diag_std_table.csv", index=False)
dt.to_csv(outdir/f"aggregated_gen_for_src_data.csv", index=False)

print("Finished all")
