"""Post-hoc analysis of the bootstrap results (no algorithm is re-run).

For every algorithm, dataset and bootstrap iteration in bootstrap_results/, the
bootstrap sample (in-bag) and out-of-bag rows are regenerated with bootstrap.py and:

    - total_selectors: number of subgroups x selectors per subgroup,
    - best_wracc_oob: WRAcc on the out-of-bag rows of the subgroup with the highest
      in-bag WRAcc (a per-subgroup descriptive metric that does not depend on how many
      subgroups a method returns),
    - for GSD, the post-hoc pruning step (neuralsd/pruning.py) removes every selector whose removal
      does not change the coverage of its subgroup on the in-bag rows. The in-bag
      Youden's index is therefore unchanged by construction, but the out-of-bag one
      may change; both the pruned description length and the pruned out-of-bag
      Youden's index are reported.

Paired differences of the out-of-bag Youden's index between GSD (after pruning) and every other
algorithm are computed per iteration (all algorithms share the same samples) and
summarized with their mean and 95% percentile interval. Bootstrap replicates are
not independent, so no classical significance test is applied to them.

The stored results of some algorithms lack the subgroup descriptions. For those, the
descriptions of the same iteration in bootstrap_results_rerun/ (rerun_descriptions.sh)
are used for the description-based metrics, while Youden's index and the number of
subgroups are always taken from the original results. GSD and beam search are
deterministic, so their re-runs reproduce the original subgroups; MCTS4DM and DiffNaps
re-runs may differ from the original ones.

Outputs:
    bootstrap_results/posthoc_iterations.csv  one row per algorithm, dataset and iteration
    bootstrap_results/posthoc_summary.csv     mean per algorithm and dataset
    bootstrap_results/paired_differences.csv  GSD minus each other algorithm

Usage:
    python posthoc_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import bootstrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuralsd"))
from pruning import prune_description as prune

ARCENE_FILE = os.path.join(bootstrap.ROOT, "useCaseArcene", "arcene_preproc.csv")
PRUNED_ALGORITHMS = ["GSD"]
RERUN_DIR = os.path.join(bootstrap.ROOT, "bootstrap_results_rerun")


def stored_descriptions(result):
    """Descriptions of a result, taken from its re-run when the original lacks them."""
    if result.get("descriptions") is not None:
        return result["descriptions"], "original"
    path = os.path.join(RERUN_DIR, result["algorithm"], result["dataset"], f"iteration_{result['iteration']}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["descriptions"], "rerun"
    return None, None


def load(dataset):
    """Load a dataset with categorical columns, which keeps Facebook small in memory."""
    if dataset == "arcene":
        df = pd.read_csv(ARCENE_FILE).astype(str)
        df.columns = [str(c) for c in df.columns]
        target = ("Target", "1")
    else:
        df, target = bootstrap.load_dataset(dataset)
    return df.astype("category"), target


def selector_matrix(df, description):
    """Boolean matrix (selectors x rows) telling which rows satisfy each selector."""
    if not description:
        return np.ones((0, len(df)), dtype=bool)
    return np.vstack([(df[attribute] == value).to_numpy() for attribute, value in description])


def wracc(df, target, description):
    covered = bootstrap.covers(df, description).to_numpy()
    positive = (df[target[0]] == target[1]).to_numpy()
    n = len(df)
    if covered.sum() == 0:
        return 0.0
    return covered.sum() / n * (positive[covered].mean() - positive.mean())


def analyze():
    results = {}
    for path in glob.glob(os.path.join(bootstrap.RESULTS_DIR, "*", "*", "iteration_*.json")):
        with open(path) as f:
            result = json.load(f)
        if result["algorithm"].endswith("permutation"):
            continue
        results.setdefault(result["dataset"], []).append(result)

    rows = []
    for dataset, dataset_results in results.items():
        df, target = load(dataset)
        samples = {}
        for result in sorted(dataset_results, key=lambda r: (r["iteration"], r["algorithm"])):
            iteration = result["iteration"]
            if iteration not in samples:
                train, test, _ = bootstrap.bootstrap_sample(df, target, iteration)
                samples[iteration] = (train, test)
            train, test = samples[iteration]
            stored, source = stored_descriptions(result)
            row = {
                "algorithm": result["algorithm"],
                "dataset": dataset,
                "iteration": iteration,
                "youden_oob": result["youden_oob"],
                "youden_in_bag": result["youden_in_bag"],
                "n_subgroups": result["n_subgroups"],
                "total_selectors": result["n_subgroups"] * (result["avg_size"] or 0),
                "best_wracc_oob": np.nan,
                "descriptions_source": source,
            }
            if stored is None:
                rows.append(row)
                continue
            descriptions = [[tuple(selector) for selector in d] for d in stored]
            if source == "rerun":
                row["rerun_youden_oob"] = bootstrap.youden(test, target, descriptions)[0]
            if descriptions:
                in_bag = [wracc(train, target, d) for d in descriptions]
                row["best_wracc_oob"] = wracc(test, target, descriptions[int(np.argmax(in_bag))])
            if result["algorithm"] in PRUNED_ALGORITHMS:
                pruned = [prune(train, d) for d in descriptions]
                row["pruned_total_selectors"] = sum(len(d) for d in pruned)
                row["pruned_avg_size"] = np.mean([len(d) for d in pruned]) if pruned else np.nan
                row["pruned_youden_in_bag"] = bootstrap.youden(train, target, pruned)[0]
                row["pruned_youden_oob"] = bootstrap.youden(test, target, pruned)[0]
            rows.append(row)
        print(f"{dataset}: {len(dataset_results)} results analyzed", flush=True)

    iterations = pd.DataFrame(rows)
    iterations.to_csv(os.path.join(bootstrap.RESULTS_DIR, "posthoc_iterations.csv"), index=False)

    summary = iterations.drop(columns="descriptions_source").groupby(["dataset", "algorithm"]).agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.drop(columns=[c for c in summary.columns if c.startswith("iteration_")])
    summary.to_csv(os.path.join(bootstrap.RESULTS_DIR, "posthoc_summary.csv"))

    paired = []
    for dataset, group in iterations.groupby("dataset"):
        # GSD is compared after its pruning step, as reported in the paper
        values = group["youden_oob"].where(group["algorithm"] != "GSD", group["pruned_youden_oob"])
        oob = group.assign(youden_oob=values).pivot(index="iteration", columns="algorithm", values="youden_oob")
        if "GSD" not in oob:
            continue
        for algorithm in oob.columns.drop("GSD"):
            difference = (oob["GSD"] - oob[algorithm]).dropna()
            paired.append({
                "dataset": dataset,
                "algorithm": algorithm,
                "iterations": len(difference),
                "mean_difference": difference.mean(),
                "ci_low": difference.quantile(0.025),
                "ci_high": difference.quantile(0.975),
                "gsd_wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
            })
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(bootstrap.RESULTS_DIR, "paired_differences.csv"), index=False)

    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    print(paired.round(3).to_string())


if __name__ == "__main__":
    analyze()
