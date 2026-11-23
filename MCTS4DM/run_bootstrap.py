# Bootstrap evaluation of MCTS4DM: runs ONE bootstrap iteration on one dataset.
#
# The bootstrap sample is written in the MCTS4DM input format (as datasets/<dataset>/preproc.ipynb
# does) and the configuration file is a copy of the original one of the dataset
# (synthetic.conf, pollutedSynthetic.conf, Facebook.conf, brcaN.conf) where only the input
# files and the result folder are changed.
#
# The labels are computed comparing the target column as a string. The preproc.ipynb notebooks
# of synthetic, polluted_synthetic and brca_n compared an integer column with "1", which
# produced qualities.csv files with no positive labels.
#
# The runtime is runtime(ms) from the info.log file written by MCTS4DM.
#
# Usage (from the MCTS4DM folder):
#     python run_bootstrap.py <dataset> <iteration>

# %%
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

sys.path.append("..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])

if bootstrap.already_done("MCTS4DM", dataset, iteration):
    print(f"MCTS4DM {dataset} iteration {iteration} already done")
    sys.exit(0)

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
original_conf = {
    "synthetic": "synthetic.conf",
    "polluted_synthetic": "pollutedSynthetic.conf",
    "Facebook": "Facebook.conf",
    "brca_n": "brcaN.conf",
}[dataset]

# Input files of this iteration
name = f"bootstrap_{dataset}_{iteration}"
input_folder = os.path.join("datasets", name)
os.makedirs(input_folder, exist_ok=True)
properties_file = os.path.join(input_folder, "properties.csv")
qualities_file = os.path.join(input_folder, "qualities.csv")
conf_file = os.path.join(input_folder, "parameters.conf")

qualities = pd.DataFrame({target[0]: (train[target[0]] == target[1]).astype(int)})
qualities.to_csv(qualities_file, index=False, sep="\t")
properties = train.drop(columns=[target[0]])
properties.columns = ['"' + c + '"' for c in properties.columns]
properties.to_csv(properties_file, index=False, sep="\t")

with open(original_conf) as f:
    conf_lines = f.readlines()
with open(conf_file, "w") as f:
    for line in conf_lines:
        if line.startswith("attrFile"):
            line = f"attrFile = {properties_file}\n"
        elif line.startswith("targetFile"):
            line = f"targetFile = {qualities_file}\n"
        elif line.startswith("resultFolderName"):
            line = f"resultFolderName = {name}\n"
        f.write(line)

# Facebook is the only dataset with boolean attributes
boolean_attributes = any(line.replace(" ", "").startswith("attrType=Boolean") for line in conf_lines)

# %%
subprocess.run(["java", "-jar", "MCTS4DM.jar", conf_file], check=True, stdout=subprocess.DEVNULL)

# MCTS4DM writes its results to results/<name>/resXP<timestamp>/
results_folder = os.path.join("results", name)
run_folder = os.path.join(results_folder, os.listdir(results_folder)[0])
results = pd.read_csv(os.path.join(run_folder, "result.log"), sep="\t")
with open(os.path.join(run_folder, "info.log")) as f:
    for line in f:
        if line.startswith("runtime(ms)"):
            time_taken = float(line.split(":")[1]) / 1000

shutil.rmtree(input_folder)
shutil.rmtree(results_folder)

# %%
# Nominal descriptions have one bracket per selector: ["""col_1""" = 4], ["""col_2""" = 0]
# Boolean descriptions are a single list of items: ["""765""", """3810"""], each item meaning value 1
descriptions = []
for description_str in results["Description"]:
    if boolean_attributes:
        items = re.findall(r'"+([^",\]]+?)"+', description_str)
        descriptions.append([(item, "1") for item in items])
    else:
        selectors = re.findall(r'\[\s*"*([^"\]]+?)"*\s*=\s*([^\]]+?)\s*\]', description_str)
        descriptions.append(selectors)

bootstrap.save_result("MCTS4DM", dataset, iteration, seed, time_taken, None, descriptions, train, test, target)
