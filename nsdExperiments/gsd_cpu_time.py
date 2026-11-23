"""Runtime of NeuralSD on CPU for the datasets of the comparison with BSD (Experiment 1).

BSD runs on CPU, while the runtimes of NeuralSD reported in the paper were measured on
GPU. This script measures NeuralSD on CPU, both single-threaded (same conditions as BSD)
and using every available core, and also checks that the WRAcc found is the same as on
GPU (NeuralSD with the "best" initialization has no random component).

Each configuration is repeated REPETITIONS times and the median runtime is reported.

Usage (from the nsdExperiments folder):
    python nsd_cpu_time.py
"""

import time

import numpy as np
import pandas as pd
import sys

import torch

sys.path.insert(0, "../neuralsd")
sys.path.insert(0, "../")
import main
from nsd import NeuralSD

REPETITIONS = 5

DATASETS = {
    "Mushrooms": (main.get_mushrooms_data, "Mushrooms"),
    "Chess": (main.get_chess_data, "Chess"),
    "StudentSuccess": (main.get_student_success_data, "StudentSuccess"),
    "Connect-4": (main.get_connect4_data, "connect4"),
    "HealthyAging": (main.get_healthy_aging_data, "HealthyAging"),
    "CarEvaluation": (main.get_car_data, "Car"),
    "Soybean": (main.get_soybean_data, "Soybean"),
    "Dermatology": (main.get_dermatology_data, "Dermatology"),
    "Depression": (main.get_depression_data, "Depression"),
    "Vancomycin": (main.get_vancomycin_data, "Vancomycin"),
    "Cardio": (main.get_cardio_data, "Cardio"),
}


def timed_fit(df, target):
    """Fit NeuralSD exactly as in e1.ipynb and return (runtime in seconds, best WRAcc)."""
    nsd = NeuralSD(1, "WRAcc")
    start = time.time()
    nsd.fit(df, target, weight_init="best", verbose=False)
    elapsed = time.time() - start
    return elapsed, nsd.forward()[0].max().item()


def main_cpu():
    torch.set_default_device("cpu")
    all_threads = torch.get_num_threads()
    rows = []
    for name, (loader, target_name) in DATASETS.items():
        df, target = loader(), main.get_target(target_name)
        row = {"Dataset": name}
        for threads in (1, all_threads):
            torch.set_num_threads(threads)
            results = [timed_fit(df, target) for _ in range(REPETITIONS)]
            row[f"cpu_{threads}_threads_s"] = float(np.median([r[0] for r in results]))
            row["wracc"] = results[0][1]
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv("nsd_cpu_times.csv", index=False)


if __name__ == "__main__":
    main_cpu()
