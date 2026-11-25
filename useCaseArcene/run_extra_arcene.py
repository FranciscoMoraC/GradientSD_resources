"""Additional ARCENE experiments requested in the review.

Two modes:

    beam         Beam search applied directly to the 26,357 selectors of the discretized
                 dataset (no dimensionality reduction), with the same bootstrap protocol as
                 run_bootstrap_arcene.py. Two quality measures are run: ChiSquaredQF (the
                 configuration selected for the PCA baseline) and StandardQF(a=0.5) (the one
                 selected for beam search in the other high-dimensional datasets); both
                 use beam width 25, 10 subgroups and depth 3. Results go to
                 ../bootstrap_results/{Beam_chi2,Beam_std05}/arcene/. The BEAM_DEPTH
                 environment variable sets another maximum depth d, and the results then go
                 to {Beam_chi2,Beam_std05}_d<d>; BEAM_PROCESSES limits the parallel processes.

    permutation  Permutation test for GradientSD. In each permutation the target is
                 shuffled over the whole dataset, a stratified bootstrap sample is drawn,
                 and GradientSD is run with the same configuration as in
                 run_bootstrap_arcene.py. The in-bag and out-of-bag Youden's indices and the
                 WRAcc of the first subgroup give their null distribution in this search
                 space. Results go to ../bootstrap_results/GSD_permutation/arcene/.

Usage (from the useCaseArcene folder):
    python run_extra_arcene.py beam
    python run_extra_arcene.py permutation [n_permutations]
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pysubgroup as ps
import torch

sys.path.insert(0, "../neuralsd")
sys.path.append("..")
import bootstrap
from iterative_gsd import IterativeGSD
from run_bootstrap_arcene import DATASET, TARGET, bootstrap_rows, load_discretized

BEAM_CONFIGS = {
    "Beam_chi2": ps.ChiSquaredQF,
    "Beam_std05": lambda: ps.StandardQF(0.5),
}
# Maximum depth of the beam search; results of a depth other than 3 get the suffix _d<depth>
BEAM_DEPTH = int(os.environ.get("BEAM_DEPTH", 3))
BEAM_SUFFIX = "" if BEAM_DEPTH == 3 else f"_d{BEAM_DEPTH}"


def run_beam_iteration(job):
    """Run one beam search configuration on one bootstrap iteration (in a worker process)."""
    df, iteration, name = job
    train_rows, test_rows, seed = bootstrap_rows(df, iteration)
    train = df.iloc[train_rows].reset_index(drop=True)
    test = df.iloc[test_rows].reset_index(drop=True)
    t0, cpu_t0 = time.time(), time.process_time()
    target = ps.BinaryTarget(TARGET[0], TARGET[1])
    search_space = ps.create_nominal_selectors(train, ignore=[TARGET[0]])
    task = ps.SubgroupDiscoveryTask(train, target, search_space, result_set_size=10,
                                    depth=BEAM_DEPTH, qf=BEAM_CONFIGS[name]())
    result = ps.BeamSearch(beam_width=25).execute(task)
    time_taken, cpu_time_taken = time.time() - t0, time.process_time() - cpu_t0
    descriptions = [
        [(sel.attribute_name, sel.attribute_value) for sel in subgroup.selectors]
        for _, subgroup in result.to_descriptions()
    ]
    bootstrap.save_result(name + BEAM_SUFFIX, DATASET, iteration, seed, time_taken, cpu_time_taken,
                          descriptions, train, test, TARGET)


def run_beam(df):
    """Beam search is single-threaded, so the iterations run in parallel processes and its
    runtime is reported as CPU time, as for the other single-threaded baselines."""
    jobs = [(df, iteration, name) for iteration in range(30) for name in BEAM_CONFIGS
            if not bootstrap.already_done(name + BEAM_SUFFIX, DATASET, iteration)]
    with Pool(min(len(jobs), int(os.environ.get("BEAM_PROCESSES", 60)))) as pool:
        pool.map(run_beam_iteration, jobs, chunksize=1)


def run_permutation(df, n_permutations):
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    for permutation in range(n_permutations):
        if bootstrap.already_done("GSD_permutation", DATASET, permutation):
            continue
        rng = np.random.default_rng([bootstrap.SEED, 1000 + permutation])
        shuffled = df.copy()
        shuffled[TARGET[0]] = rng.permutation(shuffled[TARGET[0]].to_numpy())
        train_rows, test_rows, seed = bootstrap_rows(shuffled, permutation)
        train = shuffled.iloc[train_rows].reset_index(drop=True)
        test = shuffled.iloc[test_rows].reset_index(drop=True)
        bootstrap.set_seed(seed)
        gsd = IterativeGSD(k=1, quality_measure="WRAcc")
        t0, cpu_t0 = time.time(), time.process_time()
        gsd.fit(train, TARGET, max_iterations=10, min_quality=0.07, weight_init="best",
                additional_params_for_weight_init={"n": 1}, verbose=False)
        time_taken, cpu_time_taken = time.time() - t0, time.process_time() - cpu_t0
        descriptions = [d[0] for d in gsd.subgroups]
        bootstrap.save_result("GSD_permutation", DATASET, permutation, seed, time_taken, cpu_time_taken,
                              descriptions, train, test, TARGET)


def main():
    df = load_discretized()
    if sys.argv[1] == "beam":
        run_beam(df)
    elif sys.argv[1] == "permutation":
        run_permutation(df, int(sys.argv[2]) if len(sys.argv) > 2 else 100)


if __name__ == "__main__":
    main()
