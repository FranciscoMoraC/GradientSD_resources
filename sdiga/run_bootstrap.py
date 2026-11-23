# Bootstrap evaluation of SDIGA: runs ONE bootstrap iteration on one dataset.
# The hyperparameters are the ones of run.sh, and min_confidence the one fixed in main_sdiga.py.
#
# Usage (from the sdiga folder):
#     python run_bootstrap.py <dataset> <iteration>

# %%
import os
import sys
import time

from subgroups.algorithms.subgroup_sets.sdiga import SDIGA

sys.path.append("..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])

if bootstrap.already_done("SDIGA", dataset, iteration):
    print(f"SDIGA {dataset} iteration {iteration} already done")
    sys.exit(0)

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
if dataset == "Facebook":
    confidence_weight = 0.3
    support_weight = 0.5
elif dataset == "synthetic":
    confidence_weight = 0.5
    support_weight = 0.4
elif dataset == "polluted_synthetic":
    confidence_weight = 0.3
    support_weight = 0.4
elif dataset == "brca_n":
    confidence_weight = 0.5
    support_weight = 0.4

# SDIGA writes the subgroups it finds to this file
results_file = f"sdiga_{dataset}_{iteration}.txt"

sdiga = SDIGA(max_generation=500, population_size=7000, crossover_prob=0.9, mutation_prob=0.1,
              confidence_weight=confidence_weight, support_weight=support_weight, min_confidence=0.1,
              max_pattern_size=10, num_subgroups=30, write_results_in_file=True, file_path=results_file)

# %%
t0 = time.time()
cpu_t0 = time.process_time()
sdiga.fit(train, target)
time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
# Each line of the file looks like:
# Description: [col_131 = '4', col_154 = '4'], Target: Target = '1' ; Quality Measure ...
descriptions = []
with open(results_file) as f:
    for line in f:
        description_str = line.split("Description: [")[1].split("], Target:")[0]
        description = []
        if description_str:
            for selector in description_str.split(", "):
                attribute, value = selector.split(" = ")
                description.append((attribute, value.replace("'", "")))
        descriptions.append(description)
os.remove(results_file)

bootstrap.save_result("SDIGA", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
