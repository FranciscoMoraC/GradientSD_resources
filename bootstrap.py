# Common functions for the bootstrap evaluation (<algorithm>/run_bootstrap.py scripts).
# The bootstrap sample of an iteration only depends on the dataset and the iteration,
# so all the algorithms are evaluated on the same samples.

import json
import os
import random

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("BOOTSTRAP_RESULTS_DIR", os.path.join(ROOT, "bootstrap_results"))
SEED = 42

# file in data/, target column and positive value
DATASETS = {
    "synthetic": ("synthetic3_all.csv", "Target", "1"),
    "polluted_synthetic": ("synthetic3-5_all.csv", "Target", "1"),
    "Facebook": ("facebook.csv", "target", "government"),
    "brca_n": ("brca_n.csv", "Target", "1"),
}


def load_dataset(dataset):
    file_name, target_column, target_value = DATASETS[dataset]
    df = pd.read_csv(os.path.join(ROOT, "data", file_name)).astype(str)
    df.columns = [str(c) for c in df.columns]
    return df, (target_column, target_value)


def bootstrap_sample(df, target, iteration):
    """Stratified bootstrap sample (train), out-of-bag rows (test) and a seed for the algorithm."""
    rng = np.random.default_rng([SEED, iteration])
    rows = np.arange(len(df))
    positive = (df[target[0]] == target[1]).to_numpy()
    train_rows = np.concatenate([
        rng.choice(rows[positive], size=positive.sum(), replace=True),
        rng.choice(rows[~positive], size=(~positive).sum(), replace=True),
    ])
    train_rows = rng.permutation(train_rows)
    test_rows = np.setdiff1d(rows, train_rows)
    seed = int(rng.integers(0, 2**31 - 1))
    return df.iloc[train_rows].reset_index(drop=True), df.iloc[test_rows].reset_index(drop=True), seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def result_path(algorithm, dataset, iteration):
    return os.path.join(RESULTS_DIR, algorithm, dataset, f"iteration_{iteration}.json")


def already_done(algorithm, dataset, iteration):
    return os.path.exists(result_path(algorithm, dataset, iteration))


def covers(df, description):
    """Rows covered by a subgroup given as a list of (attribute, value) selectors."""
    entry = pd.Series(True, index=df.index)
    for attribute, value in description:
        entry &= df[attribute] == value
    return entry


def youden(df, target, descriptions):
    """Youden's index, TPR and FPR of the disjunction of the subgroups."""
    covered = pd.Series(False, index=df.index)
    for description in descriptions:
        covered |= covers(df, description)
    positive = df[target[0]] == target[1]
    tpr = (covered & positive).sum() / positive.sum()
    fpr = (covered & ~positive).sum() / (~positive).sum()
    return float(tpr - fpr), float(tpr), float(fpr)


def max_jaccard(df, descriptions):
    entries = [covers(df, description) for description in descriptions]
    maximum = 0.0
    for i in range(len(entries)):
        for j in range(i):
            union = (entries[i] | entries[j]).sum()
            if union > 0:
                maximum = max(maximum, (entries[i] & entries[j]).sum() / union)
    return float(maximum)


def save_result(algorithm, dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target):
    """Compute Youden's index in-bag (train) and out-of-bag (test) and save the result as JSON."""
    youden_train, tpr_train, fpr_train = youden(train, target, descriptions)
    youden_test, tpr_test, fpr_test = youden(test, target, descriptions)
    sizes = [len(description) for description in descriptions]
    result = {
        "algorithm": algorithm,
        "dataset": dataset,
        "iteration": iteration,
        "seed": seed,
        "time_s": time_taken,
        "cpu_time_s": cpu_time_taken,
        "youden_in_bag": youden_train,
        "tpr_in_bag": tpr_train,
        "fpr_in_bag": fpr_train,
        "youden_oob": youden_test,
        "tpr_oob": tpr_test,
        "fpr_oob": fpr_test,
        "n_subgroups": len(descriptions),
        "avg_size": float(np.mean(sizes)) if sizes else None,
        "max_jaccard": max_jaccard(train, descriptions),
        "n_in_bag": len(train),
        "n_oob": len(test),
        "descriptions": [[[attribute, value] for attribute, value in description] for description in descriptions],
    }
    path = result_path(algorithm, dataset, iteration)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{algorithm} {dataset} {iteration}: time={time_taken:.1f}s Youden in-bag={youden_train:.4f} OOB={youden_test:.4f}")
