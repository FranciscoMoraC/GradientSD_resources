# Bootstrap evaluation of GradientSD (same hyperparameters as run_<dataset>.py).
# Usage: python run_bootstrap.py <dataset> <iteration>

# %%
import sys
import time

import torch

sys.path.insert(0, "../gradientsd")
from iterative_gsd import IterativeGSD

sys.path.append("..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])

if bootstrap.already_done("GSD", dataset, iteration):
    sys.exit(0)

if torch.cuda.is_available():
    torch.set_default_device("cuda:0")

k = 1

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
if dataset == "synthetic":
    gsd = IterativeGSD(k, "Qc", additional_params_for_quality_measure={"c": 0.9})
    fit_params = {"max_iterations": 10, "min_quality": 10}
elif dataset == "polluted_synthetic":
    gsd = IterativeGSD(k, "Qc", additional_params_for_quality_measure={"c": 0.9})
    fit_params = {"max_iterations": 10, "min_quality": 10}
elif dataset == "Facebook":
    gsd = IterativeGSD(k, "WRAcc")
    fit_params = {"batch_size": 1024, "max_iterations": 10, "min_quality": 0.01}
elif dataset == "brca_n":
    gsd = IterativeGSD(k, "Qc", additional_params_for_quality_measure={"c": 0.5})
    fit_params = {"max_iterations": 10, "min_quality": 10}

# %%
t0 = time.time()
cpu_t0 = time.process_time()
gsd.fit(train, target, weight_init="best", verbose=False, **fit_params)
time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
descriptions = [d[0] for d in gsd.get_subgroup_descriptions()]

bootstrap.save_result("GSD", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
