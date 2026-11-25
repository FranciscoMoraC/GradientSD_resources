# Grid search of beam search on the discovery split (80%) of a dataset.
# In pysubgroup the number of subgroups only trims the final beam, so we run each
# (measure, beam width, depth) once keeping the top 50 subgroups, and select_hyperparams.py
# takes the top 3, 5, 10, 20 and 50 from the stored results.
#
# Usage: python run_hyperparam.py <dataset>

# %%
import os
import random
import sys
from itertools import product

import pysubgroup as ps

sys.path.append("..")
import main

dataset = sys.argv[1]
measures = {"chisquaredQF": ps.ChiSquaredQF(), "standardQF_0.5": ps.StandardQF(0.5), "standardQF_1.0": ps.StandardQF(1.0)}
beam_widths = [10, 25, 100, 250]
depths = [3, 5]
os.makedirs("hyperparam_tuning_results", exist_ok=True)

# %%
main.params["split"] = True
data, _ = main.get_data(database=dataset)
target_tuple = main.get_target(database=dataset)

cols = data.columns.tolist()
random.shuffle(cols)
data = data[cols]
target = ps.BinaryTarget(target_tuple[0], target_tuple[1])
search_space = ps.create_nominal_selectors(data, ignore=[target_tuple[0]])

# %%
for (measure, qf), beam_width, depth in product(measures.items(), beam_widths, depths):
    file = f"hyperparam_tuning_results/{dataset}_depth{depth}_beamsize{beam_width}_{measure}.csv"
    if os.path.exists(file):
        continue
    task = ps.SubgroupDiscoveryTask(data, target, search_space, result_set_size=min(beam_width, 50), depth=depth, qf=qf)
    result = ps.BeamSearch(beam_width=beam_width).execute(task)
    result.to_dataframe().to_csv(file, index=False)
    print(file)
