from pathlib import Path
import pandas as pd

filepath = Path(__file__).parent

infer_dir = filepath / "out_infer"
preds_fname = "test_y_data_predicted.csv"

canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"


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
    ff = df.groupby(['improve_sample_id', 'improve_chem_id']).agg(
        auc_pred_mean=('auc_pred', 'mean'),
        auc_std_mean=('auc_pred', 'std'))
    ff = ff.reset_index().sort_values(['improve_sample_id',
                                       'improve_chem_id',
                                       'auc_pred_mean'])
    return ff


# CTRP
source = "CTRPv2"
target = "PDMR"
res_dir = infer_dir / f"{source}-{target}"
cc = agg_pred_files(res_dir)
ca = agg_pred_scores(cc)


# GDSC
source = "GDSCv2"
target = "PDMR"
res_dir = infer_dir / f"{source}-{target}"
gg = agg_pred_files(res_dir)
ga = agg_pred_scores(gg)


# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Values': [10, 15, 14, 22, 21, 23, 30, 31, 29]
}
df = pd.DataFrame(data)

# Calculate mean and standard deviation
means = df.groupby('Category')['Values'].mean()
stds = df.groupby('Category')['Values'].std()

# Create a bar plot with error bars
means.plot(kind='bar', yerr=stds, capsize=4)
plt.ylabel('Values')
plt.title('Mean and Standard Deviation of Values by Category')
plt.show()

print("Finished")
