import os
from pathlib import Path
import pandas as pd

fdir = Path(__file__).resolve().parent

data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
fea_list = ["ge", "mordred"]
fea_sep = "_"
trg_name = "AUC"
seed = 0

def groupby_src_and_print(df, print_fn=print):
    print_fn(df.groupby('SOURCE').agg({'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())

# src = "ccle"
# src = "ctrp"
# src = "gcsi"
# src = "gdsc1"
# src = "gdsc2"
# datadir = fdir/f"ml.dfs/July2020/data.{src}"

datadir = fdir/"ml.dfs/July2020"


def load_and_merge(src, use_lincs=True, verbose=False):
    """ ... """
    src_dir = datadir/f"data.{src}"
    splitdir = src_dir/"splits"

    # Load data
    rsp = pd.read_csv(src_dir/f"rsp_{src}.csv")      # Drug response
    ge = pd.read_csv(src_dir/f"ge_{src}.csv")        # Gene expressions
    mrd = pd.read_csv(src_dir/f"mordred_{src}.csv")  # Mordred descriptors
    fps = pd.read_csv(src_dir/f"ecfp2_{src}.csv")    # Morgan fingerprints
    smi = pd.read_csv(src_dir/f"smiles_{src}.csv")   # SMILES

    # Use landmark genes
    if use_lincs:
        with open(src_dir/"../landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        genes = ["ge_" + str(g) for g in genes]
        print(len(set(genes).intersection(set(ge.columns[1:]))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = ["CancID"] + genes
        ge = ge[cols]

    if verbose:
        groupby_src_and_print(rsp)
        print("Unique cell lines with gene expressions", ge["CancID"].nunique())
        print("Unique drugs with Mordred", mrd["DrugID"].nunique())
        print("Unique drugs with ECFP2", fps["DrugID"].nunique())

    # Merge (tidy df)
    data = pd.merge(rsp, ge, on='CancID', how='inner')
    data = pd.merge(data, mrd, on='DrugID', how='inner')
    groupby_src_and_print(data)
    return data


# Get features (x), target (y), and meta
def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]

    
def split_data(data, src, split):
    """ ... """
    src_dir = datadir/f"data.{src}"
    splitdir = src_dir/"splits"
    # print("\nGet the splits.")

    with open(splitdir/f"split_{split}_tr_id") as f:
        tr_id = [int(line.rstrip()) for line in f]

    with open(splitdir/f"split_{split}_te_id") as f:
        te_id = [int(line.rstrip()) for line in f]

    # Train and test data
    tr_data = data.loc[tr_id]
    te_data = data.loc[te_id]

    # Val data from tr_data
    from sklearn.model_selection import train_test_split
    tr_data, vl_data = train_test_split(tr_data, test_size=0.12, random_state=seed)

    tr_data = tr_data.reset_index(drop=True)
    vl_data = vl_data.reset_index(drop=True)
    te_data = te_data.reset_index(drop=True)
    print("Train", tr_data.shape)
    print("Val  ", vl_data.shape)
    print("Test ", te_data.shape)

    xtr, ytr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep), tr_data[[trg_name]]
    xvl, yvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep), vl_data[[trg_name]]
    xte, yte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep), te_data[[trg_name]]
    assert xtr.shape[0] == ytr.shape[0], "Size missmatch."
    assert xvl.shape[0] == yvl.shape[0], "Size missmatch."
    assert xte.shape[0] == yte.shape[0], "Size missmatch."

    return tr_data, vl_data, te_data, xtr, ytr, xvl, yvl, xte, yte


# Scores
def calc_scores(y_true, y_pred):
    """ ... """
    import sklearn
    from scipy.stats import pearsonr, spearmanr
    scores = {}
    scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
    scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
    return scores


# import ipdb; ipdb.set_trace(context=5)
for src in data_sources:

    for split in range(10):

        data = load_and_merge(src, use_lincs=True)
        tr_data, vl_data, te_data, xtr, ytr, xvl, yvl, xte, yte = split_data(data, src, split)

        # Scale
        # from ml.scale import scale_fea
        # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

        # Train model
        import lightgbm as lgb
        ml_init_args = {'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.1,
                        'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
        ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
        ml_fit_args['eval_set'] = (xvl, yvl)
        model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
        model.fit(xtr, ytr, **ml_fit_args)

        # outdir = fdir/f"results.csa.{src}"
        outdir = fdir/f"results.csa"
        os.makedirs(outdir, exist_ok=True)

        # Predict
        y_pred = model.predict(xte)
        y_true = yte.values.squeeze()
        pred = pd.DataFrame({"True": y_true, "Pred": y_pred})
        te_df = te_data[["DrugID", "CancID", trg_name]]
        pred = pd.concat([te_df, pred], axis=1)
        assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
       
        pred_fname = f"{src}_{src}_split_{split}.csv"
        pred.to_csv(outdir/pred_fname, index=False)

        # Drop preds on target set
        trg_studies = [s for s in data_sources if s not in src]
        # import ipdb; ipdb.set_trace(context=5)
        for trg_study in trg_studies:

            data = load_and_merge(trg_study, use_lincs=True)
            xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
            ydata = data[[trg_name]]
            meta = data[["DrugID", "CancID", trg_name]]

            y_pred = model.predict(xdata)
            y_true = ydata.values.squeeze()
            pred = pd.DataFrame({"True": y_true, "Pred": y_pred})

            pred = pd.concat([meta, pred], axis=1)
            assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns trg_name and 'True' are the ground truth."
            pred_fname = f"{src}_{trg_study}_split_{split}.csv"
            pred.to_csv(outdir/pred_fname, index=False)

            scores = calc_scores(y_true=y_true, y_pred=y_pred)
            print(scores)
