"""Out-of-bag bootstrap evaluation of NeuralSD and of the PCA + beam search baseline on ARCENE.

Both methods are evaluated with the same protocol as the high-dimensional experiment:
for each iteration, a stratified bootstrap sample (same size as the dataset) is drawn,
the subgroups are discovered on it, and Youden's index of their disjunction is computed
both on the bootstrap sample (in-bag) and on the rows not drawn (out-of-bag).
Every method sees exactly the same samples (see ../bootstrap.py).
The results are written to ../bootstrap_results/{NSD,PCA_Beam}/arcene/.

Hyperparameters are the ones used for the single-run results reported in the paper:
    - NeuralSD: WRAcc, at most 10 subgroups, minimum quality 0.07 (run.ipynb).
    - PCA + beam search: best configuration of pca_beam_search.py
      (ChiSquaredQF, beam width 25, 10 subgroups, depth 3).
For the baseline, PCA and the quantile discretization are fitted on the bootstrap
sample only and then applied to the out-of-bag rows.

Usage (from the useCaseArcene folder):
    python run_bootstrap_arcene.py [n_iterations]
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pysubgroup as ps
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, "../neuralsd")
sys.path.append("..")
import bootstrap
from iterative_nsd import IterativeNSD

DATASET = "arcene"
TARGET = ("Target", "1")
N_ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def load_discretized():
    """ARCENE discretized in four quantile bins (data_preproc.ipynb), as strings."""
    df = pd.read_csv("arcene_preproc.csv").astype(str)
    df.columns = [str(c) for c in df.columns]
    return df


def load_raw():
    """Raw (continuous) ARCENE training + validation data, row-aligned with arcene_preproc.csv."""
    train = pd.read_csv("ARCENE/arcene_train.data", header=None, sep=r"\s+")
    valid = pd.read_csv("ARCENE/arcene_valid.data", header=None, sep=r"\s+")
    return pd.concat([train, valid], ignore_index=True)


def bootstrap_rows(df, iteration):
    """Row indices of the bootstrap sample and of the out-of-bag rows of an iteration.

    Uses the same random generator as bootstrap.bootstrap_sample, so the samples
    match the ones of the rest of the bootstrap experiments for a given iteration.
    """
    rng = np.random.default_rng([bootstrap.SEED, iteration])
    rows = np.arange(len(df))
    positive = (df[TARGET[0]] == TARGET[1]).to_numpy()
    train_rows = np.concatenate([
        rng.choice(rows[positive], size=positive.sum(), replace=True),
        rng.choice(rows[~positive], size=(~positive).sum(), replace=True),
    ])
    train_rows = rng.permutation(train_rows)
    test_rows = np.setdiff1d(rows, train_rows)
    seed = int(rng.integers(0, 2**31 - 1))
    return train_rows, test_rows, seed


def run_neuralsd(df, train_rows, test_rows, seed, iteration):
    if bootstrap.already_done("NSD", DATASET, iteration):
        return
    train = df.iloc[train_rows].reset_index(drop=True)
    test = df.iloc[test_rows].reset_index(drop=True)
    bootstrap.set_seed(seed)
    nsd = IterativeNSD(k=1, quality_measure="WRAcc")
    t0, cpu_t0 = time.time(), time.process_time()
    nsd.fit(train, TARGET, max_iterations=10, min_quality=0.07, weight_init="best",
            additional_params_for_weight_init={"n": 1}, verbose=False)
    time_taken, cpu_time_taken = time.time() - t0, time.process_time() - cpu_t0
    descriptions = [d[0] for d in nsd.subgroups]
    bootstrap.save_result("NSD", DATASET, iteration, seed, time_taken, cpu_time_taken,
                          descriptions, train, test, TARGET)


def run_pca_beam(df, raw, train_rows, test_rows, seed, iteration):
    if bootstrap.already_done("PCA_Beam", DATASET, iteration):
        return
    t0, cpu_t0 = time.time(), time.process_time()
    # PCA and discretization are fitted on the bootstrap sample only
    pca = PCA()
    train_pca = pca.fit_transform(raw.iloc[train_rows])
    test_pca = pca.transform(raw.iloc[test_rows])
    discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    train = pd.DataFrame(discretizer.fit_transform(train_pca)).astype(str)
    test = pd.DataFrame(discretizer.transform(test_pca)).astype(str)
    train.columns = test.columns = [str(c) for c in train.columns]
    train[TARGET[0]] = df[TARGET[0]].iloc[train_rows].to_numpy()
    test[TARGET[0]] = df[TARGET[0]].iloc[test_rows].to_numpy()

    target = ps.BinaryTarget(TARGET[0], TARGET[1])
    search_space = ps.create_nominal_selectors(train, ignore=[TARGET[0]])
    task = ps.SubgroupDiscoveryTask(train, target, search_space, result_set_size=10,
                                    depth=3, qf=ps.ChiSquaredQF())
    result = ps.BeamSearch(beam_width=25).execute(task)
    time_taken, cpu_time_taken = time.time() - t0, time.process_time() - cpu_t0
    descriptions = [
        [(sel.attribute_name, sel.attribute_value) for sel in subgroup.selectors]
        for _, subgroup in result.to_descriptions()
    ]
    bootstrap.save_result("PCA_Beam", DATASET, iteration, seed, time_taken, cpu_time_taken,
                          descriptions, train, test, TARGET)


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    df = load_discretized()
    raw = load_raw()
    for iteration in range(N_ITERATIONS):
        train_rows, test_rows, seed = bootstrap_rows(df, iteration)
        run_neuralsd(df, train_rows, test_rows, seed, iteration)
        run_pca_beam(df, raw, train_rows, test_rows, seed, iteration)


if __name__ == "__main__":
    main()
