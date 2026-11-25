# Grid search of the PCA + discretization + beam search baseline on ARCENE (whole dataset).

# %%
from itertools import product

import pandas as pd
import pysubgroup as ps
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

df = pd.concat([pd.read_csv("ARCENE/arcene_train.data", header=None, sep=r"\s+"),
                pd.read_csv("ARCENE/arcene_valid.data", header=None, sep=r"\s+")], ignore_index=True)
labels = pd.concat([pd.read_csv("ARCENE/arcene_train.labels", header=None, sep=r"\s+"),
                    pd.read_csv("arcene_valid.labels", header=None, sep=r"\s+")], ignore_index=True)

pca_dataset = pd.DataFrame(KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
                           .fit_transform(PCA().fit_transform(df)))
pca_dataset["Target"] = labels[0]
pca_dataset = pca_dataset.astype(str)
target_tuple = ("Target", "1")


def youden(data, descriptions):
    """Youden's index of the disjunction of the subgroups (pysubgroup string descriptions)."""
    positive = data[target_tuple[0]] == target_tuple[1]
    covered = pd.Series(False, index=data.index)
    for description in descriptions:
        entry = pd.Series(True, index=data.index)
        for selector in description.split(" AND "):
            attr, value = selector.split("==")
            entry &= data[int(attr.strip())] == value.strip().strip("'").strip('"')
        covered |= entry
    return (covered & positive).sum() / positive.sum() - (covered & ~positive).sum() / (~positive).sum()


# %%
measures = {"ChiSquaredQF": ps.ChiSquaredQF(), "StandardQF_0.5": ps.StandardQF(0.5), "StandardQF_1.0": ps.StandardQF(1.0)}
beam_widths = [10, 25, 100, 250]
n_subgroups = [3, 5, 10, 20, 50]
depths = [3, 5]

target = ps.BinaryTarget(*target_tuple)
search_space = ps.create_nominal_selectors(pca_dataset, ignore=[target_tuple[0]])
records = []
for (measure, qf), beam_width, num_subgroups, depth in product(measures.items(), beam_widths, n_subgroups, depths):
    if num_subgroups > beam_width:
        continue
    task = ps.SubgroupDiscoveryTask(pca_dataset, target, search_space, result_set_size=num_subgroups, depth=depth, qf=qf)
    result = ps.BeamSearch(beam_width=beam_width).execute(task)
    descriptions = [str(d) for d in result.to_dataframe()["subgroup"]]
    records.append({"measure": measure, "beam_width": beam_width, "n_subgroups": num_subgroups,
                    "depth": depth, "youden": youden(pca_dataset, descriptions)})

# %%
results = pd.DataFrame(records)
results.to_csv("youden_results_pca_beam_search.csv", index=False)
print(results.sort_values("youden", ascending=False).head(10).to_string(index=False))
