# Bootstrap evaluation of Beam Search (pysubgroup): runs ONE bootstrap iteration on one dataset.
# The hyperparameters are the ones selected in run.sh and the search depth is the one of run.py.
#
# Usage (from the beam folder):
#     python run_bootstrap.py <dataset> <iteration>

# %%
import random
import sys
import time

import pysubgroup as ps

sys.path.append("..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])

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

# %%
t0 = time.time()
cpu_t0 = time.process_time()

# As in run.py, the order of the columns is shuffled before the search
cols = train.columns.tolist()
random.shuffle(cols)
train = train[cols]

ps_target = ps.BinaryTarget(target[0], target[1])
search_space = ps.create_nominal_selectors(train, ignore=[target[0]])
task = ps.SubgroupDiscoveryTask(train, ps_target, search_space, result_set_size=num_subgroups, depth=3, qf=quality_measure)
result = ps.BeamSearch(beam_width=beam_width).execute(task)

time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
# result.results holds (quality, subgroup, statistics) tuples
descriptions = []
for _, subgroup, *_ in result.results:
    descriptions.append([(selector.attribute_name, selector.attribute_value) for selector in subgroup.selectors])

bootstrap.save_result("Beam", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
