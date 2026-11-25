# Hyperparameter search of Beam Search (pysubgroup) on the discovery split (80%) of a dataset,
# with the maximum depth as an additional hyperparameter.
#
# In pysubgroup, the result set size only trims the final beam (the search itself depends on
# the beam width, the quality measure and the depth), so a single run keeps the top
# min(beam_width, 50) subgroups, and the top-k subgroups for every k of the grid are the
# first k rows of the stored file (sorted by quality). select_hyperparams_depth.py then
# evaluates every configuration on the validation split (20%).
#
# Usage (from the beam folder):
#     python run_hyperparam_depth.py <dataset> <measure> <beam_width> <depth>

import os
import random
import sys

import pysubgroup as ps

sys.path.append("..")
import main

dataset = sys.argv[1]
measure = sys.argv[2]
beam_width = int(sys.argv[3])
depth = int(sys.argv[4])

# The largest number of subgroups of the grid is 50
MAX_SUBGROUPS = 50
OUTPUT_DIR = "hyperparam_tuning_results_depth"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, f"{dataset}_depth{depth}_beamsize{beam_width}_{measure}.csv")
if os.path.exists(output_file):
    print(f"{output_file} already done")
    sys.exit(0)

if measure.startswith("standardQF"):
    quality_measure = ps.StandardQF(float(measure.split("_")[1]))
elif measure == "chisquaredQF":
    quality_measure = ps.ChiSquaredQF()

# Discovery split only: the validation split is used for the selection
main.params["split"] = True
train, _ = main.get_data(database=dataset)
target_tuple = main.get_target(database=dataset)

# As in run_hyperparam.py, the order of the columns is shuffled before the search
cols = train.columns.tolist()
random.shuffle(cols)
train = train[cols]

target = ps.BinaryTarget(target_tuple[0], target_tuple[1])
search_space = ps.create_nominal_selectors(train, ignore=[target_tuple[0]])
task = ps.SubgroupDiscoveryTask(train, target, search_space,
                                result_set_size=min(beam_width, MAX_SUBGROUPS), depth=depth, qf=quality_measure)
result = ps.BeamSearch(beam_width=beam_width).execute(task)

# to_dataframe returns the subgroups sorted by decreasing quality
result.to_dataframe().to_csv(output_file, index=False)
print(f"done {output_file}")
