# Runtime of GradientSD on CPU (one thread and all threads, median of 5 runs) on the datasets
# of Experiment 1, checking that the WRAcc is the same as on GPU.

import time

import numpy as np
import pandas as pd
import sys

import torch

sys.path.insert(0, "../gradientsd")
sys.path.insert(0, "../")
import main
from gsd import GradientSD

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
    """Fit GradientSD exactly as in e1.ipynb and return (runtime in seconds, best WRAcc)."""
    gsd = GradientSD(1, "WRAcc")
    start = time.time()
    gsd.fit(df, target, weight_init="best", verbose=False)
    elapsed = time.time() - start
    return elapsed, gsd.forward()[0].max().item()


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
    pd.DataFrame(rows).to_csv("gsd_cpu_times.csv", index=False)


if __name__ == "__main__":
    main_cpu()
