# Bootstrap evaluation of beam search (pysubgroup) with the hyperparameters in selected_hyperparams.csv.
# Usage: python run_bootstrap.py <dataset> <iteration>

# %%
import random
import sys
import time

import pandas as pd
import pysubgroup as ps

sys.path.append("..")
import bootstrap

dataset = sys.argv[1]
iteration = int(sys.argv[2])
if bootstrap.already_done("Beam", dataset, iteration):
    sys.exit(0)

config = pd.read_csv("selected_hyperparams.csv").set_index("Dataset").loc[dataset]
measure = config["Quality measure"]
quality_measure = ps.ChiSquaredQF() if measure == "chisquaredQF" else ps.StandardQF(float(measure.split("_")[1]))

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
t0 = time.time()
cpu_t0 = time.process_time()
cols = train.columns.tolist()
random.shuffle(cols)
train = train[cols]
ps_target = ps.BinaryTarget(target[0], target[1])
search_space = ps.create_nominal_selectors(train, ignore=[target[0]])
task = ps.SubgroupDiscoveryTask(train, ps_target, search_space, result_set_size=int(config["Number of subgroups"]),
                                depth=int(config["Depth"]), qf=quality_measure)
result = ps.BeamSearch(beam_width=int(config["Beam width"])).execute(task)
time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
descriptions = [[(s.attribute_name, s.attribute_value) for s in subgroup.selectors] for _, subgroup, *_ in result.results]
bootstrap.save_result("Beam", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
