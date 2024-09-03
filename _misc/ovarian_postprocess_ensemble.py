""" Compute ensemble of model predictions. """
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = Path(__file__).parent  # py
# filepath = Path(os.path.abspath(''))  # ipynb
print(filepath)

plots_outdir = filepath / 'plots_outdir'

# Models dir
canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"

# -----------------------------------------
datadir = filepath / "plots_outdir"
target = "PDMR"
# -----------------------------------------

# IMPROVE models
source = "CTRPv2"
improve_models_list = ["DeepTTC", "GraphDRP", "HIDRA", "IGTD", "PaccMann_MCA"]
dfs = []
for model_name in improve_models_list:
    agg_df = pd.read_csv(filepath / "plots_outdir" / f"agg_preds_{model_name}_{source}_{target}.tsv", sep="\t")
    agg_df["model"] = model_name
    dfs.append(agg_df)
df_improve = pd.concat(dfs, axis=0)
print(df_improve.shape)

# UNO
source = "all"
model_name = "UNO"
df_uno = pd.read_csv(filepath / "plots_outdir" / f"agg_preds_{model_name}_{source}_{target}.tsv", sep="\t")
df_uno["model"] = model_name
print(df_uno.shape)

df = pd.concat([df_improve, df_uno], axis=0)
print(df.shape)


pdo_t = "655913~031-T"
pdo_r = "937885~149-R"

tt = df[df[canc_col_name].isin([pdo_t])].reset_index(drop=True)
rr = df[df[canc_col_name].isin([pdo_r])].reset_index(drop=True)

# group_by_cols = [canc_col_name, drug_col_name]
# ff = tt.groupby(group_by_cols).agg(
#     pred_mean=(y_col_name, 'mean'),
#     pred_std=(y_col_name, 'std'))
# ff = ff.reset_index().sort_values([drug_col_name, canc_col_name, 'pred_mean']).reset_index(drop=True)

tt_piv = tt.pivot(index=['drug_name'], columns=['model'], values=['pred_mean']).reset_index()
model_name_cols = [i[1] for i in tt_piv.columns.values[1:]]
tt_piv.columns = ['drug_name'] + model_name_cols
tt_piv['Avg'] = tt_piv[model_name_cols].mean(axis=1)
# tt_piv = tt_piv.sort_values('Avg').reset_index()
tt_piv = tt_piv.sort_values('drug_name').reset_index(drop=True)


df = tt_piv.drop(columns=['Avg'])
df.to_csv(plots_outdir / "table_drug_by_model_auc.tsv", sep="\t", index=False)
ranked_df = df.set_index("drug_name")
ranked_df = ranked_df.rank().astype(int)  # compute ranked from raw auc values
ranked_df.to_csv(plots_outdir / "table_drug_by_model_ranked.tsv", sep="\t", index=True)

# rank_corr = ranked_df.corr(method='spearman', axis=0)

v = df.loc[0, model_name_cols].values
corr, pvalue = spearmanr(v, v)

df.set_index('drug_name', inplace=True)
tran_df = df.transpose()


def calc_ensemble(df: pd.DataFrame):
    df_piv = df.pivot(index=['drug_name'], columns=['model'], values=['pred_mean']).reset_index()
    model_name_cols = [i[1] for i in df_piv.columns.values[1:]]
    df_piv.columns = ['drug_name'] + model_name_cols
    # df_piv = df_piv.set_index('drug_name')
    # df_piv['Avg'] = df_piv.mean(axis=1)
    tt_piv['Avg'] = tt_piv[model_name_cols].mean(axis=1)
    df_piv = df_piv.sort_values('Avg').reset_index(drop=True)
    return df_div


tt_ens = calc_ensemble(tt)
print(tt_ens)

rr_ens = calc_ensemble(rr)
print(rr_ens)


# Aggregate predictions
# For regression or probability predictions
ensemble_pred = np.mean([pred1, pred2, pred3], axis=0)

# For classification with majority voting (assuming predictions are class labels)
ensemble_pred_labels = np.argmax(np.bincount([np.argmax(pred1, axis=1), np.argmax(pred2, axis=1), np.argmax(pred3, axis=1)], axis=0))

print(ensemble_pred)
print(ensemble_pred_labels)



print("Finished")


