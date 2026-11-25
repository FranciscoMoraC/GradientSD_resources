# Youden's index on the validation split (20%) of every configuration of the grid search
# (run_hyperparam.py) and selection of the best one per dataset.

# %%
import os
import re
import sys

import pandas as pd

sys.path.append("..")
import main

NUM_SUBGROUPS = [3, 5, 10, 20, 50]
main.params["split"] = True


def youden(data, positive, descriptions):
    covered = pd.Series(False, index=data.index)
    for description in descriptions:
        entry = pd.Series(True, index=data.index)
        for selector in description.split(" AND "):
            attr, value = selector.split("==")
            entry &= data[attr] == value.strip().strip("'").strip('"')
        covered |= entry
    return (covered & positive).sum() / positive.sum() - (covered & ~positive).sum() / (~positive).sum()


# %%
validation = {}
rows = []
for file in sorted(os.listdir("hyperparam_tuning_results")):
    match = re.match(r"(.+)_depth(\d+)_beamsize(\d+)_(.+)\.csv$", file)
    if match is None:
        continue
    dataset, depth, width, measure = match[1], int(match[2]), int(match[3]), match[4]
    if dataset not in validation:
        _, data = main.get_data(database=dataset)
        target = main.get_target(database=dataset)
        validation[dataset] = (data, data[target[0]] == target[1])
    data, positive = validation[dataset]
    subgroups = pd.read_csv(os.path.join("hyperparam_tuning_results", file))["subgroup"].tolist()
    for k in NUM_SUBGROUPS:
        if k <= width:
            rows.append([dataset, measure, width, k, depth, youden(data, positive, subgroups[:k])])

# %%
results = pd.DataFrame(rows, columns=["Dataset", "Quality measure", "Beam width", "Number of subgroups", "Depth", "Youden"])
results.to_csv("hyperparam_tuning_results/all_configurations.csv", index=False)
selected = results.loc[results.groupby("Dataset")["Youden"].idxmax()]
selected.to_csv("selected_hyperparams.csv", index=False)
print(selected.to_string(index=False))
