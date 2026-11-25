# Bootstrap evaluation of Beam Search (pysubgroup): runs ONE bootstrap iteration on one dataset.
# The hyperparameters are the ones selected in run.sh and the search depth is the one of run.py
# (3), unless the BEAM_DEPTH environment variable sets a different maximum depth or BEAM_CONFIG
# points to the configuration selected by select_hyperparams_depth.py.
#
# Usage (from the beam folder):
#     python run_bootstrap.py <dataset> <iteration>

# %%
import os
import random
import sys
import time

import pysubgroup as ps

sys.path.append("..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])
# Maximum number of selectors per subgroup
depth = int(os.environ.get("BEAM_DEPTH", 3))

if bootstrap.already_done("Beam", dataset, iteration):
    print(f"Beam {dataset} iteration {iteration} already done")
    sys.exit(0)

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
if dataset == "Facebook":
    quality_measure = ps.StandardQF(1.0)
    beam_width = 100
    num_subgroups = 3
elif dataset == "brca_n":
    quality_measure = ps.StandardQF(0.5)
    beam_width = 25
    num_subgroups = 10
elif dataset == "polluted_synthetic":
    quality_measure = ps.StandardQF(0.5)
    beam_width = 100
    num_subgroups = 20
elif dataset == "synthetic":
    quality_measure = ps.StandardQF(0.5)
    beam_width = 250
    num_subgroups = 50

# The BEAM_CONFIG environment variable may point to a file written by
# select_hyperparams_depth.py, whose configuration (including the depth) replaces the one above
config_file = os.environ.get("BEAM_CONFIG")
if config_file:
    import pandas as pd
    config = pd.read_csv(config_file).set_index("Dataset").loc[dataset]
    measure = config["Quality measure"]
    quality_measure = ps.ChiSquaredQF() if measure == "chisquaredQF" else ps.StandardQF(float(measure.split("_")[1]))
    beam_width = int(config["Beam width"])
    num_subgroups = int(config["Number of subgroups"])
    depth = int(config["Depth"])
    print(f"Beam {dataset}: {measure}, beam width {beam_width}, {num_subgroups} subgroups, depth {depth}")

# %%
t0 = time.time()
cpu_t0 = time.process_time()

# As in run.py, the order of the columns is shuffled before the search
cols = train.columns.tolist()
random.shuffle(cols)
train = train[cols]

ps_target = ps.BinaryTarget(target[0], target[1])
search_space = ps.create_nominal_selectors(train, ignore=[target[0]])
task = ps.SubgroupDiscoveryTask(train, ps_target, search_space, result_set_size=num_subgroups, depth=depth, qf=quality_measure)
result = ps.BeamSearch(beam_width=beam_width).execute(task)

time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
# result.results holds (quality, subgroup, statistics) tuples
descriptions = []
for _, subgroup, *_ in result.results:
    descriptions.append([(selector.attribute_name, selector.attribute_value) for selector in subgroup.selectors])

bootstrap.save_result("Beam", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
