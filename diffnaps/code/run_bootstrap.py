# Bootstrap evaluation of DiffNaps (same configurations as conf_dict in real_exp.py).
# Usage: python run_bootstrap.py <dataset> <iteration>

# %%
import sys
import time

import numpy as np

from method.diffnaps import DiffnapsNet, learn_diffnaps_net
from utils.utils_base import TrainConfig, get_positional_patterns

sys.path.append("../..")
import bootstrap

# %%
dataset = sys.argv[1]
iteration = int(sys.argv[2])

if bootstrap.already_done("DiffNaps", dataset, iteration):
    sys.exit(0)

# %%
df, target = bootstrap.load_dataset(dataset)
train, test, seed = bootstrap.bootstrap_sample(df, target, iteration)
bootstrap.set_seed(seed)

# %%
if dataset == "Facebook":
    conf = TrainConfig(hidden_dim=2000, epochs=100, weight_decay=10.0, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                       lambda_c=25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=32,
                       log_interval=100, sparse_regu=0, test=False, lr=0.001, model=DiffnapsNet, seed=14401360119984179300, init_enc="bimodal")
elif dataset == "synthetic":
    conf = TrainConfig(hidden_dim=2000, epochs=25, weight_decay=5.0, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                       lambda_c=25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=32,
                       log_interval=100, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet, seed=14401360119984179300, init_enc="bimodal")
elif dataset == "polluted_synthetic":
    conf = TrainConfig(hidden_dim=500, epochs=50, weight_decay=5.0, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                       lambda_c=25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=128,
                       log_interval=100, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, seed=14401360119984179300, init_enc="bimodal")
elif dataset == "brca_n":
    conf = TrainConfig(hidden_dim=10000, epochs=100, weight_decay=10.0, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                       lambda_c=25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=64,
                       log_interval=100, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet, seed=14401360119984179300, init_enc="bimodal")
conf.t1 = 0.03
conf.t2 = 0.8

# %%
t0 = time.time()
cpu_t0 = time.process_time()

# DiffNaps works with a numeric matrix and integer labels
labels = (train[target[0]] == target[1]).to_numpy().astype(int)
data = train.drop(columns=[target[0]]).astype(float).to_numpy().astype(np.float32)
columns = [c for c in train.columns if c != target[0]]

model, new_weights, trainDS = learn_diffnaps_net(data, conf, labels=labels)
enc_w = model.fc0_enc.weight.data.detach().cpu()
c_w = model.classifier.weight.detach().cpu()
# Extract the differential patterns, t1 is t_e and t2 is t_c (as in real_exp.py)
_, _, _, num_pat, res_dict, _ = get_positional_patterns(enc_w, c_w, general=True, t_mean=1.0, t1=conf.t1, t2=conf.t2)
res_dict = {int(k): v for k, v in res_dict.items()}

time_taken = time.time() - t0
cpu_time_taken = time.process_time() - cpu_t0

# %%
# Keep the patterns of the positive class (label 1) that res_to_csv (utils/experiment_utils.py)
# writes to the .tsv files used by the original Youden notebooks: patterns covering more than
# one row with a rounded precision for the class above 0%
patterns = []
for pattern in res_dict.get(1, []):
    covered = np.sum(data[:, pattern], axis=1) == len(pattern)
    hit = labels[covered] == 1
    if len(hit) > 1 and round(np.sum(hit) / len(hit) * 100) > 0:
        patterns.append(pattern)

# A pattern covers a row when all its items are present (value 1, written as "1.0" in brca_n)
one = "1.0" if dataset == "brca_n" else "1"
descriptions = [[(columns[i], one) for i in pattern] for pattern in patterns]

bootstrap.save_result("DiffNaps", dataset, iteration, seed, time_taken, cpu_time_taken, descriptions, train, test, target)
