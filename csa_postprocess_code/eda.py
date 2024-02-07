from pathlib import Path
import pandas as pd

fdir = Path(__file__).resolve().parent

# src = "ccle"
src = "ctrp"
# src = "gcsi"
# src = "gdsc1"
# src = "gdsc2"

datadir = fdir/f"ml.dfs/July2020/data.{src}"
splitdir = datadir/"splits"

def groupby_src_and_print(df, print_fn=print):
    print_fn(df.groupby('SOURCE').agg({'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())

# Load data
rsp = pd.read_csv(datadir/f"rsp_{src}.csv")      # Drug response
ge = pd.read_csv(datadir/f"ge_{src}.csv")        # Gene expressions
mrd = pd.read_csv(datadir/f"mordred_{src}.csv")  # Mordred descriptors
fps = pd.read_csv(datadir/f"ecfp2_{src}.csv")    # Morgan fingerprints
smi = pd.read_csv(datadir/f"smiles_{src}.csv")   # SMILES

# Use landmark genes
use_lincs = True
if use_lincs:
    with open(datadir/"../landmark_genes") as f:
        genes = [str(line.rstrip()) for line in f]
    genes = ["ge_" + str(g) for g in genes]
    print(len(set(genes).intersection(set(ge.columns[1:]))))
    genes = list(set(genes).intersection(set(ge.columns[1:])))
    cols = ["CancID"] + genes
    ge = ge[cols]

groupby_src_and_print(rsp)
print("Unique cell lines with gene expressions", ge["CancID"].nunique())
print("Unique drugs with Mordred", mrd["DrugID"].nunique())
print("Unique drugs with ECFP2", fps["DrugID"].nunique())

# Merge (tidy df)
data = pd.merge(rsp, ge, on='CancID', how='inner')
data = pd.merge(data, mrd, on='DrugID', how='inner')
groupby_src_and_print(data)
del rsp, ge, mrd, fps, smi


# -----------------------------------------------
#   Train model
# -----------------------------------------------
# Example of training a DRP model with gene expression and Mordred descriptors
print("\nGet the splits.")

with open(splitdir/"split_0_tr_id") as f:
    tr_id = [int(line.rstrip()) for line in f]

with open(splitdir/"split_0_te_id") as f:
    te_id = [int(line.rstrip()) for line in f]

# Train and test data
tr_data = data.loc[tr_id]
te_data = data.loc[te_id]

# Val data from tr_data
from sklearn.model_selection import train_test_split
tr_data, vl_data = train_test_split(tr_data, test_size=0.12)
print("Train", tr_data.shape)
print("Val  ", vl_data.shape)
print("Test ", te_data.shape)
del data

# Get features (x), target (y), and meta
def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
fea_list = ["ge", "mordred"]
fea_sep = "_"
trg_name = "AUC"
xtr, ytr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep), tr_data[[trg_name]]
xvl, yvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep), vl_data[[trg_name]]
xte, yte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep), te_data[[trg_name]]
assert xtr.shape[0] == ytr.shape[0], "Size missmatch."
assert xvl.shape[0] == yvl.shape[0], "Size missmatch."
assert xte.shape[0] == yte.shape[0], "Size missmatch."
del tr_data, vl_data, te_data

# Scale
# from ml.scale import scale_fea
# xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

# Train model
# import ipdb; ipdb.set_trace(context=5)
import lightgbm as lgb
ml_init_args = {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1,
                'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
# ml_fit_args = {'verbose': False, 'early_stopping_rounds': 10}
ml_fit_args = {'verbose': True, 'early_stopping_rounds': 50}
ml_fit_args['eval_set'] = (xvl, yvl)
model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
model.fit(xtr, ytr, **ml_fit_args)

# Predict
y_pred = model.predict(xte)
y_true = yte.values.squeeze()

# Scores
import sklearn
from scipy.stats import pearsonr, spearmanr
scores = {}
scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
scores['mean_absolute_error']   = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
print(scores)
