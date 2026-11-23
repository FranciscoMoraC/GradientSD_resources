"""
Grid search over pysubgroup beam search parameters on PCA-transformed ARCENE data.
Computes the Youden index for each (measure, beam_width, n_subgroups) combination
and saves all results to "youden_results_pca_beam_search.csv".
"""

import pysubgroup as ps
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from itertools import product

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df_train = pd.read_csv("ARCENE/arcene_train.data", header=None, delim_whitespace=True)
df_valid = pd.read_csv("ARCENE/arcene_valid.data", header=None, delim_whitespace=True)
df = pd.concat([df_train, df_valid], ignore_index=True)

labels_train = pd.read_csv("ARCENE/arcene_train.labels", header=None, delim_whitespace=True)
labels_valid = pd.read_csv("arcene_valid.labels", header=None, delim_whitespace=True)
labels = pd.concat([labels_train, labels_valid], ignore_index=True)

df["Target"] = labels
target_tuple = ("Target", "1")

# ---------------------------------------------------------------------------
# 2. PCA + discretization (done once, shared across all grid cells)
# ---------------------------------------------------------------------------
pca = PCA()
transformed_data = pca.fit_transform(df.drop(columns=["Target"]))

pca_dataset = pd.DataFrame(
    KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    .fit_transform(transformed_data)
)
pca_dataset["Target"] = df["Target"]
pca_dataset = pca_dataset.astype(str)

# ---------------------------------------------------------------------------
# 3. Helper: compute Youden index from a list of subgroup descriptions
# ---------------------------------------------------------------------------
def compute_youden(pca_dataset: pd.DataFrame, descriptions: list[list[str]], target_tuple: tuple) -> float:
    """
    Compute the Youden index (sensitivity + specificity - 1) for the union of
    the discovered subgroups.

    Parameters
    ----------
    pca_dataset   : dataset with string-typed columns including the target.
    descriptions  : list of subgroups, each represented as a list of
                    "attr == val" selector strings.
    target_tuple  : (column_name, positive_value) pair.

    Returns
    -------
    Youden index as a float.
    """
    target_col, target_val = target_tuple
    target_row = pca_dataset[target_col] == target_val

    # Build the union mask over all subgroups
    global_entry = pd.Series(False, index=pca_dataset.index)
    for subgroup in descriptions:
        entry = pd.Series(True, index=pca_dataset.index)
        for selector in subgroup:
            attr, val = selector.split("==")
            val = val.strip().strip("'").strip('"')
            attr = eval(attr.strip())           # column index (int)
            entry = entry & (pca_dataset[attr] == val)
        global_entry = global_entry | entry

    sensitivity = sum(global_entry & target_row) / sum(target_row)
    false_positive_rate = sum(global_entry & ~target_row) / sum(~target_row)
    return sensitivity - false_positive_rate


# ---------------------------------------------------------------------------
# 4. Grid search parameters
# ---------------------------------------------------------------------------
measures = {
    "ChiSquaredQF":   ps.ChiSquaredQF(),
    "StandardQF_0.5": ps.StandardQF(0.5),
    "StandardQF_1.0": ps.StandardQF(1.0),
}

beam_widths   = [10, 25, 100, 250]
n_subgroups   = [3, 5, 10, 20, 50]

# Pre-build the search space (same for every run)
target = ps.BinaryTarget(target_tuple[0], target_tuple[1])
search_space = ps.create_nominal_selectors(pca_dataset, ignore=[target_tuple[0]])

# ---------------------------------------------------------------------------
# 5. Run grid search
# ---------------------------------------------------------------------------
records = []
total = len(measures) * len(beam_widths) * len(n_subgroups)
run = 0

for (measure_name, quality_measure), beam_width, num_subgroups in product(
    measures.items(), beam_widths, n_subgroups
):
    run += 1
    print(
        f"[{run}/{total}] measure={measure_name}, beam_width={beam_width}, "
        f"n_subgroups={num_subgroups} ...",
        end=" ",
        flush=True,
    )

    if num_subgroups > beam_width:
        print("SKIPPED (n_subgroups > beam_width)")
        continue

    # Build and execute the task
    task = ps.SubgroupDiscoveryTask(
        pca_dataset,
        target,
        search_space,
        result_set_size=num_subgroups,
        depth=3,
        qf=quality_measure,
    )
    result = ps.BeamSearch(beam_width=beam_width).execute(task)
    results_df = result.to_dataframe()

    # Parse subgroup descriptions into lists of selectors
    descriptions = [
        [s for s in str(desc).split(" AND ")]
        for desc in results_df["subgroup"].to_list()
    ]

    # Compute Youden index
    youden = compute_youden(pca_dataset, descriptions, target_tuple)
    print(f"Youden = {youden:.4f}")

    records.append(
        {
            "measure":      measure_name,
            "beam_width":   beam_width,
            "n_subgroups":  num_subgroups,
            "youden":       youden,
        }
    )

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
output_df = pd.DataFrame(records)
output_path = "youden_results_pca_beam_search.csv"
output_df.to_csv(output_path, index=False)
print(f"\nResults saved to '{output_path}'.")
print(output_df.sort_values("youden", ascending=False).to_string(index=False))