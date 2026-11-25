# Experiment 1: WRAcc of the best single selector (initial subgroup) and of GradientSD, number of
# epochs, and sensitivity to the learning rate, the patience and the initial weight magnitude.
# Usage (from a folder whose ../data has the datasets): python ablation_e1.py

import itertools
import os
import sys
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "neuralsd"))
sys.path.insert(0, os.path.join(HERE, ".."))
import main
from gsd import GradientSD

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

LEARNING_RATES = [1, 10, 100]
PATIENCES = [2, 10]
INIT_VALUES = [5, 10, 20]


def discrete_wracc(X, T, mask):
    """WRAcc of the conjunction of the selectors in mask.

    X is the (selectors x instances) binary matrix, T the binary target (instances,)
    and mask a boolean vector over the selectors.
    """
    covered = X[mask].prod(dim=0) if mask.any() else torch.ones_like(T)
    n = T.numel()
    tp = (covered * T).sum()
    cov = covered.sum()
    if cov == 0:
        return 0.0
    return float(cov / n * (tp / cov - T.sum() / n))


def single_selector_wracc(X, T):
    """Discrete WRAcc of the best subgroup described by a single selector.

    X is the (selectors x instances) binary matrix and T the binary target (instances,).
    """
    n = T.numel()
    cov = X.sum(dim=1)
    tp = X @ T
    wracc = cov / n * (tp / cov.clamp(min=1) - T.sum() / n)
    return float(wracc.max())


def run_gsd(df, target, lr=100, patience=2, init_value=10):
    """Fit GradientSD and return its relaxed and discrete WRAcc, epochs and runtime."""
    gsd = GradientSD(1, "WRAcc")
    t0 = time.time()
    gsd.fit(df, target, weight_init="best", additional_params_for_weight_init={"value": init_value},
            lr=lr, patience=patience, verbose=False)
    elapsed = time.time() - t0
    X = gsd._selector_matrix.matrix.double()
    T = gsd.target[0].double()
    mask = (torch.sigmoid(gsd.W[:, 0]) > 0.5)
    return {
        "relaxed_wracc": float(gsd.forward()[0]),
        "discrete_wracc": discrete_wracc(X, T, mask),
        "single_selector_wracc": single_selector_wracc(X, T),
        "initial_wracc": float(gsd.initial_quality[0]),
        "epochs": gsd.epochs_run,
        "time_s": elapsed,
    }, X, T


def main_ablation():
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    rows, sensitivity = [], []
    for name, (loader, target_name) in DATASETS.items():
        df, target = loader(), main.get_target(target_name)
        default, _, _ = run_gsd(df, target)
        row = {"dataset": name, **default}
        rows.append(row)
        print(row, flush=True)
        for lr, patience, init_value in itertools.product(LEARNING_RATES, PATIENCES, INIT_VALUES):
            result, _, _ = run_gsd(df, target, lr, patience, init_value)
            sensitivity.append({"dataset": name, "lr": lr, "patience": patience, "init_value": init_value, **result})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "ablation_e1.csv"), index=False)
    pd.DataFrame(sensitivity).to_csv(os.path.join(HERE, "ablation_e1_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main_ablation()
