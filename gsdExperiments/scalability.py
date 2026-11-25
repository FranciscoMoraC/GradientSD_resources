"""Scalability of GradientSD with respect to the number of rows (n) and attributes (p).

The data follow the generative structure of the Synthetic dataset of Experiment 2:
six informative binary attributes whose nine combinations keep the exact proportions of
the original dataset (three disjoint pure subgroups define the positive class), plus p noise
attributes with 5 uniformly distributed values. GradientSD uses the same configuration as
in the bootstrap evaluation of Synthetic (Qc with c = 0.9, "best" initialization, at most
10 subgroups, full batch).

Each call runs ONE configuration, so that the peak memory of the process is not affected
by previous runs, and appends one JSON line to the results file.

Usage (from the gsdExperiments folder):
    python scalability.py <n> <p> <repetition> <results_file>
"""

import json
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "../gradientsd")
from iterative_gsd import IterativeGSD

# Combinations of (Target, A1, ..., A6) in the original Synthetic dataset and their counts
PATTERNS = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0],
])
COUNTS = np.array([500, 500, 500, 500, 500, 500, 1000, 1500, 1500])
NOISE_VALUES = 5
TARGET = ("Target", "1")


def generate(n, p, seed):
    """Generate a Synthetic-like dataset with n rows and p noise attributes (as strings).

    The informative part keeps the exact counts of the original dataset scaled to n (n must
    be a multiple of 3500), so the seed only changes the noise attributes and the row order.
    """
    if ((n * COUNTS) % COUNTS.sum()).any():
        raise ValueError("n must be a multiple of 3500 to keep the exact proportions")
    rng = np.random.default_rng(seed)
    informative = rng.permutation(np.repeat(PATTERNS, n * COUNTS // COUNTS.sum(), axis=0))
    noise = rng.integers(0, NOISE_VALUES, size=(n, p))
    columns = ["Target"] + [f"Attribute {i}" for i in range(1, 7)] + [f"col_{i}" for i in range(1, p + 1)]
    df = pd.DataFrame(np.hstack([informative, noise]), columns=columns)
    return df.astype(str)


def youden(df, descriptions):
    """Youden's index of the disjunction of the discovered subgroups on df."""
    covered = pd.Series(False, index=df.index)
    for description in descriptions:
        entry = pd.Series(True, index=df.index)
        for attribute, value in description:
            entry &= df[attribute] == value
        covered |= entry
    positive = df[TARGET[0]] == TARGET[1]
    return float((covered & positive).sum() / positive.sum() - (covered & ~positive).sum() / (~positive).sum())


def main():
    n, p, repetition, results_file = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seed = 1000 * repetition + 7
    df = generate(n, p, seed)
    torch.manual_seed(seed)

    # Initialize the CUDA context before timing, so that it is not part of the runtime
    torch.set_default_device("cuda:0")
    torch.zeros(1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    result = {"n": n, "p": p, "selectors": None, "repetition": repetition}
    gsd = IterativeGSD(1, "Qc", additional_params_for_quality_measure={"c": 0.9})
    try:
        t0 = time.time()
        cpu_t0 = time.process_time()
        gsd.fit(df, TARGET, max_iterations=10, weight_init="best", verbose=False)
        torch.cuda.synchronize()
        result["time_s"] = time.time() - t0
        result["cpu_time_s"] = time.process_time() - cpu_t0
        descriptions = [d[0] for d in gsd.get_subgroup_descriptions()]
        result["subgroups"] = len(descriptions)
        result["youden"] = youden(df, descriptions)
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
    # Number of selectors: values of every non-target attribute
    result["selectors"] = int(sum(df[c].nunique() for c in df.columns if c != TARGET[0]))
    result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    # ru_maxrss is given in KiB on Linux; it includes the generated DataFrame
    result["rss_peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10

    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
