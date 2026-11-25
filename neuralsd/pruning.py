"""Post-hoc pruning of subgroup descriptions found by GradientSD.

GradientSD has no incentive to drop selectors that barely change the coverage (e.g.,
absence conditions on sparse binary attributes that most instances satisfy), so its
descriptions may contain redundant selectors. This step removes every selector whose
removal does not change the set of instances covered by the subgroup on the discovery
data. The in-sample coverage, and hence every in-sample quality measure, is unchanged.
"""

import numpy as np


def prune_description(df, description):
    """Remove the selectors of a description whose removal does not change its coverage on df.

    df is a pandas DataFrame and description a list of (attribute, value) selectors.
    fails[v] counts the selectors of the current description that row v does not satisfy,
    so v is covered iff fails[v] == 0; removing selector j changes the coverage iff some
    row fails only j. Selectors satisfied by more rows (less informative) are tried first.
    """
    if not description:
        return list(description)
    satisfied = np.vstack([(df[attribute] == value).to_numpy() for attribute, value in description])
    fails = (~satisfied).sum(axis=0)
    keep = np.ones(len(description), dtype=bool)
    for j in np.argsort(-satisfied.sum(axis=1), kind="stable"):
        if not np.any((fails == 1) & ~satisfied[j]):
            keep[j] = False
            fails -= ~satisfied[j]
    return [selector for selector, k in zip(description, keep) if k]


def prune_subgroups(df, descriptions):
    """Prune every description of a subgroup set on the discovery data df."""
    return [prune_description(df, description) for description in descriptions]
