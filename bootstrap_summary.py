# Summary of the bootstrap evaluation: mean, standard deviation and 95% percentile interval
# of each metric, per algorithm and dataset, over the iterations in bootstrap_results/.
#
# The runtime used to compare the algorithms is the CPU time for Beam and SDIGA, whose
# iterations run in parallel (so their wall-clock time is distorted), and the wall-clock
# time for the rest.
#
# Usage:
#     python bootstrap_summary.py

# %%
import glob
import json
import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap_results")
CPU_TIME_ALGORITHMS = ["Beam", "SDIGA"]
METRICS = ["youden_in_bag", "youden_oob", "runtime_s", "n_subgroups", "avg_size", "max_jaccard"]

# %%
rows = []
for path in glob.glob(os.path.join(RESULTS_DIR, "*", "*", "iteration_*.json")):
    with open(path) as f:
        result = json.load(f)
    result.pop("descriptions", None)
    rows.append(result)
results = pd.DataFrame(rows)

uses_cpu_time = results["algorithm"].isin(CPU_TIME_ALGORITHMS)
results["runtime_s"] = results["time_s"].where(~uses_cpu_time, results["cpu_time_s"])

# %%
summary = results.groupby(["dataset", "algorithm"])[METRICS].agg(
    ["mean", "std", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
summary.columns = [f"{metric}_{stat}" for metric in METRICS for stat in ["mean", "std", "ci_low", "ci_high"]]
summary.insert(0, "iterations", results.groupby(["dataset", "algorithm"]).size())
summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))

# %%
table = summary[["iterations", "youden_in_bag_mean", "youden_oob_mean", "youden_oob_std", "runtime_s_mean", "runtime_s_std"]]
print(table.round(3).to_string())
