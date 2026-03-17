
import sys
import os
import json
from pathlib import Path

import pandas as pd

from crowdkit.aggregation import MajorityVote, DawidSkene, GLAD

sys.path.append("./methods")
from bds_stan_wrapper import BDS
from hsds_em import HSDS_EM
from hsds_stan import HSDS_Stan

N_ITERATIONS = 10000
N_SAMPLES = 1000
N_WARMUP = 500
INIT_WORKER_ACCURACY = 0.7

def map_labels(df, label_mapping):
    df["label"] = df["label"].map(label_mapping)
    return df

def reverse_label_mapping(maped_df, label_mapping):
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    maped_df["label"] = maped_df["label"].map(reverse_mapping)
    return maped_df

# Check the arguments and load the data
if len(sys.argv) not in [5,6]:
    print("Usage: python main.py <method_name> <output_filename> <labels_json_filename> <human_filename> <ai_filename (optional)>")
    sys.exit(1)
method_name = sys.argv[1]
output_filename = sys.argv[2]
labels_json_filename = sys.argv[3]
try:
    with open("datasets/" + labels_json_filename, "r") as f:
        labels_strings = json.load(f)
        label_mapping = {label: idx for idx, label in enumerate(labels_strings)}
except Exception as e:
    print(f"Error loading labels JSON file: {e}")
    sys.exit(1)
human_filename = sys.argv[4]
try:
    human_df = pd.read_csv("datasets/" + human_filename)
except Exception as e:
    print(f"Error loading human responses CSV file: {e}")
    sys.exit(1)
if len(sys.argv) == 5:
    ai_df = None
    print("No AI responses provided, proceeding with human labels only.")
else:
    ai_filename = sys.argv[5]
    try:
        ai_df = pd.read_csv("datasets/" + ai_filename)
    except Exception as e:
        print(f"Error loading AI responses CSV file: {e}")
        sys.exit(1)

# label mapping
try:
    human_df = map_labels(human_df, label_mapping)
    if ai_df is not None:
        ai_df = map_labels(ai_df, label_mapping)
except Exception as e:
    print(f"Error occurred while mapping labels: {e}")
    print("Please check the labels JSON file and ensure it contains the correct mappings for all labels in the datasets.")
    sys.exit(1)

# Run the selected method
is_separated  = False
labels = list(label_mapping.values())

infer_params = {
    "iter_warmup": N_WARMUP,
    "iter_sampling": N_SAMPLES,
}

if method_name == "MV":
    model = MajorityVote()
elif method_name == "DS":
    model = DawidSkene(n_iter=N_ITERATIONS)
elif method_name == "GLAD":
    model = GLAD(n_iter=N_ITERATIONS)
elif method_name == "BDS":
    model = BDS(labels=labels, 
                algorithm="mcmc", 
                infer_params=infer_params,
                init_worker_accuracy=INIT_WORKER_ACCURACY)
elif method_name == "HSDS_EM":
    model = HSDS_EM(n_iter=N_ITERATIONS,
                    r=INIT_WORKER_ACCURACY)
    is_separated = True
elif method_name == "HSDS_MCMC":
    model = HSDS_Stan(labels=labels, 
                    algorithm="mcmc", 
                    infer_params=infer_params,
                    init_worker_accuracy=INIT_WORKER_ACCURACY)
    is_separated = True
else:
    print("Invalid method name. Choose from: MV, DS, GLAD, BDS, HSDS_EM, HSDS_MCMC")
    sys.exit(1)

if ai_df is None and is_separated:
    print("Error: The selected method requires both human and AI responses. Please provide an AI responses CSV file.")
    sys.exit(1)
try:
    if not is_separated:
        df = pd.concat([human_df, ai_df], axis=0) if ai_df is not None else human_df
        predicted_labels = model.fit_predict(df)
    else:
        predicted_labels = model.fit_predict(human_df, ai_df)
except Exception as e:
    print(f"Error during {method_name} fitting and prediction: {e}")
    sys.exit(1)

# reverse label mapping
try:
    predicted_labels = predicted_labels.to_frame().reset_index()
    predicted_labels.columns = ["task", "label"]
    predicted_labels = reverse_label_mapping(predicted_labels, label_mapping)
except Exception as e:
    print(f"Error occurred while reversing label mapping: {e}")
    sys.exit(1)

# Save the results
try:
    predicted_labels.to_csv("results/" + output_filename, index=False)
    print(f"Predicted labels saved to results/{output_filename}")
except Exception as e:
    print(f"Error occurred while saving results: {e}")
    sys.exit(1)

exit(0)
