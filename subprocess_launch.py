"""
To run this script:
1. Activate conda env with: source activate myenv (e.g., source activate dh)
2. Run the script: python subprocess_main.py
"""
import json
import pprint
import subprocess

# ---------------------
# Some IMPROVE settings
# ---------------------
source = "CCLE"
split = 0
train_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
val_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
model_outdir = f"out_models_dh_hpo/{source}/split_{split}"
# log_dir = "dh_hpo_logs/"
subprocess_bashscript = "subprocess_train.sh"

subprocess_res = subprocess.run(
    [
        "bash", subprocess_bashscript,
        str(train_ml_data_dir),
        str(val_ml_data_dir),
        str(model_outdir)
    ], 
    capture_output=True, text=True, check=True
)

print(subprocess_res.stdout)
print(subprocess_res.stderr)

f = open(model_outdir + "/val_scores.json")
val_scores = json.load(f)
objective = -val_scores["val_loss"]
print("objective:", objective)

breakpoint()
print("\nFinished subprocess_main.py")
