# Selects the Beam Search hyperparameters (quality measure, beam width, number of subgroups
# and maximum depth) with the best Youden's index on the validation split, from the runs of
# run_hyperparam_depth.py. The top-k subgroups of a configuration are the first k rows of
# its file, for every k in the grid not larger than the beam width.
#
# Usage (from the beam folder):
#     python select_hyperparams_depth.py
# Writes hyperparam_tuning_results_depth/all_configurations.csv and selected_hyperparams_depth.csv.

import os
import re
import sys

import pandas as pd

sys.path.append("..")
import main

INPUT_DIR = "hyperparam_tuning_results_depth"
NUM_SUBGROUPS = [3, 5, 10, 20, 50]
FILE_PATTERN = re.compile(r"^(?P<dataset>.+)_depth(?P<depth>\d+)_beamsize(?P<width>\d+)_(?P<measure>.+)\.csv$")

main.params["split"] = True
validation_cache = {}


def validation_data(dataset):
    """Returns the validation split and the boolean vector of its positive instances."""
    if dataset not in validation_cache:
        _, validation = main.get_data(database=dataset)
        target_attr, target_value = main.get_target(database=dataset)
        validation_cache[dataset] = (validation, validation[target_attr] == target_value)
    return validation_cache[dataset]


def youden(data, positives, descriptions):
    """Youden's index of the disjunction of the subgroups (pysubgroup string descriptions)."""
    covered = pd.Series(False, index=data.index)
    for description in descriptions:
        entry = pd.Series(True, index=data.index)
        for selector in description.split(" AND "):
            attr, value = selector.split("==")
            entry &= data[attr] == value.strip().strip("'").strip('"')
        covered |= entry
    return (covered & positives).sum() / positives.sum() - (covered & ~positives).sum() / (~positives).sum()


rows = []
for file in sorted(os.listdir(INPUT_DIR)):
    match = FILE_PATTERN.match(file)
    if match is None:
        continue
    dataset, measure = match["dataset"], match["measure"]
    depth, width = int(match["depth"]), int(match["width"])
    subgroups = pd.read_csv(os.path.join(INPUT_DIR, file))["subgroup"].tolist()
    data, positives = validation_data(dataset)
    for k in NUM_SUBGROUPS:
        if k <= width:
            rows.append([dataset, measure, width, k, depth, youden(data, positives, subgroups[:k])])

results = pd.DataFrame(rows, columns=["Dataset", "Quality measure", "Beam width", "Number of subgroups", "Depth", "Youden"])
results.to_csv(os.path.join(INPUT_DIR, "all_configurations.csv"), index=False)
selected = results.loc[results.groupby("Dataset")["Youden"].idxmax()]
selected.to_csv("selected_hyperparams_depth.csv", index=False)
print(results.groupby(["Dataset", "Depth"])["Youden"].max().unstack())
print(selected.to_string(index=False))
