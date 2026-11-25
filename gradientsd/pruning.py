# Post-hoc pruning of GradientSD subgroups: removes the selectors whose removal does not
# change the instances covered by the subgroup on the discovery data.

import numpy as np


def prune_description(df, description):
    """Prune a description (list of (attribute, value) selectors) on df.

    fails[v] is the number of selectors not satisfied by row v, so a selector can be
    removed if no row fails only that selector. The selectors satisfied by more rows
    are tried first.
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
    return [prune_description(df, description) for description in descriptions]
