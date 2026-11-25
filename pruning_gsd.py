# Applies the pruning step to the GradientSD subgroups stored in bootstrap_results/ and
# computes the paired differences of the out-of-bag Youden's index between GradientSD
# (after pruning) and the other algorithms.
# Some GradientSD results were stored without the subgroups; they are taken from
# bootstrap_results_rerun/ (GradientSD is deterministic).
#
# Output: bootstrap_results/gsd_pruned.csv and bootstrap_results/paired_differences.csv

# %%
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import bootstrap

sys.path.insert(0, os.path.join(bootstrap.ROOT, "neuralsd"))
from pruning import prune_subgroups

RERUN_DIR = os.path.join(bootstrap.ROOT, "bootstrap_results_rerun")


def load(dataset):
    if dataset == "arcene":
        df = pd.read_csv(os.path.join(bootstrap.ROOT, "useCaseArcene", "arcene_preproc.csv")).astype(str)
        df.columns = [str(c) for c in df.columns]
        return df, ("Target", "1")
    return bootstrap.load_dataset(dataset)


def descriptions_of(result):
    if result["descriptions"] is None:
        path = os.path.join(RERUN_DIR, "GSD", result["dataset"], f"iteration_{result['iteration']}.json")
        with open(path) as f:
            result = json.load(f)
    return [[tuple(selector) for selector in d] for d in result["descriptions"]]


# %%
rows = []
for dataset_dir in sorted(glob.glob(os.path.join(bootstrap.RESULTS_DIR, "GSD", "*"))):
    dataset = os.path.basename(dataset_dir)
    df, target = load(dataset)
    for path in sorted(glob.glob(os.path.join(dataset_dir, "iteration_*.json"))):
        with open(path) as f:
            result = json.load(f)
        train, test, _ = bootstrap.bootstrap_sample(df, target, result["iteration"])
        pruned = prune_subgroups(train, descriptions_of(result))
        rows.append({
            "dataset": dataset,
            "iteration": result["iteration"],
            "youden_oob": result["youden_oob"],
            "pruned_youden_in_bag": bootstrap.youden(train, target, pruned)[0],
            "pruned_youden_oob": bootstrap.youden(test, target, pruned)[0],
            "avg_size": result["avg_size"],
            "pruned_avg_size": np.mean([len(d) for d in pruned]),
        })
    print(dataset, "done")
pruned = pd.DataFrame(rows)
pruned.to_csv(os.path.join(bootstrap.RESULTS_DIR, "gsd_pruned.csv"), index=False)
print(pruned.groupby("dataset").mean().drop(columns="iteration").round(3))

# %%
paired = []
gsd = pruned.set_index(["dataset", "iteration"])["pruned_youden_oob"]
for path in glob.glob(os.path.join(bootstrap.RESULTS_DIR, "*", "*", "iteration_*.json")):
    with open(path) as f:
        result = json.load(f)
    if result["algorithm"] in ("GSD", "GSD_permutation"):
        continue
    key = (result["dataset"], result["iteration"])
    if key in gsd.index:
        paired.append({"dataset": key[0], "algorithm": result["algorithm"], "difference": gsd[key] - result["youden_oob"]})
paired = pd.DataFrame(paired).groupby(["dataset", "algorithm"])["difference"].agg(
    mean="mean", ci_low=lambda x: x.quantile(0.025), ci_high=lambda x: x.quantile(0.975),
    gsd_wins=lambda x: int((x > 0).sum()), ties=lambda x: int((x == 0).sum()))
paired.to_csv(os.path.join(bootstrap.RESULTS_DIR, "paired_differences.csv"))
print(paired.round(3))
