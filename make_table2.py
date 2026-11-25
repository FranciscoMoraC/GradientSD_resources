"""Build the body of Table 2 of the paper (LaTeX) from the bootstrap summaries.

Reads bootstrap_results/summary.csv (bootstrap_summary.py) and
bootstrap_results/posthoc_summary.csv (posthoc_analysis.py). GradientSD (GSD) is
reported after its post-hoc pruning step (neuralsd/pruning.py): its out-of-bag Youden's
index and its selectors per subgroup are those of the pruned subgroups, while its
runtime is that of the discovery (pruning takes milliseconds).

Usage:
    python make_table2.py                     # print the table body
    python make_table2.py --paper main.tex    # replace the table body in the paper
"""

import os
import sys

import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap_results")
ALGORITHMS = ["MCTS4DM", "DiffNaps", "Beam", "SDIGA", "GSD"]
DATASETS = [("Facebook", "Facebook"), ("synthetic", "Synthetic"),
            ("polluted_synthetic", "Noisy Synthetic"), ("brca_n", "Brca\\_n")]
# Line of the paper right before the table body, and runtime cell annotated with a footnote
TABLE_HEADER = "\\textbf{SDIGA} & \\textbf{GradientSD} \\\\ \\hline\n"
FOOTNOTE_CELL = ("& 105.1$\\pm$235.7 \\\\", "& 105.1$\\pm$235.7$^\\dagger$ \\\\")

summary = pd.read_csv(os.path.join(RESULTS, "summary.csv")).set_index(["dataset", "algorithm"])
posthoc = pd.read_csv(os.path.join(RESULTS, "posthoc_summary.csv")).set_index(["dataset", "algorithm"])


def stat(dataset, algorithm, metric, statistic):
    """Mean or std of a metric; for GSD, Youden's index and size are taken after pruning."""
    if algorithm == "GSD" and metric in ("youden_oob", "avg_size"):
        pruned = {"youden_oob": "pruned_youden_oob", "avg_size": "pruned_avg_size"}[metric]
        return posthoc.loc[(dataset, algorithm), f"{pruned}_{statistic}"]
    return summary.loc[(dataset, algorithm), f"{metric}_{statistic}"]


def mean_std(dataset, algorithm, metric, digits):
    return f"{stat(dataset, algorithm, metric, 'mean'):.{digits}f}$\\pm${stat(dataset, algorithm, metric, 'std'):.{digits}f}"


def block(title, render, best=None):
    """Rows of one block of the table; the best rounded value of each row is in bold."""
    lines = [f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\"]
    for dataset, name in DATASETS:
        cells = [render(dataset, algorithm) for algorithm in ALGORITHMS]
        if best is not None:
            values = [best(dataset, algorithm) for algorithm in ALGORITHMS]
            top = max(values) if best.__name__ == "higher" else min(values)
            cells = [f"\\textbf{{{c}}}" if v == top else c for c, v in zip(cells, values)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")
    return lines


def higher(dataset, algorithm):
    return round(stat(dataset, algorithm, "youden_oob", "mean"), 2)


def lower(dataset, algorithm):
    return round(stat(dataset, algorithm, "runtime_s", "mean"), 1)


def size(dataset, algorithm):
    n = stat(dataset, algorithm, "n_subgroups", "mean")
    n = f"{n:.0f}" if float(n).is_integer() else f"{n:.1f}"
    return f"{n} ({stat(dataset, algorithm, 'avg_size', 'mean'):.1f})"


def table_body():
    lines = []
    lines += block("Youden's index (out-of-bag)", lambda d, a: mean_std(d, a, "youden_oob", 2), higher)
    lines += block("Runtime (s)", lambda d, a: mean_std(d, a, "runtime_s", 1), lower)
    lines += block("Number of subgroups (selectors per subgroup)", size)
    return "\n".join(lines[:-1]) + " \\hline\n"


def update_paper(path):
    """Replace the body of Table 2 (between its header row and \\end{tabular}) in the paper."""
    with open(path) as f:
        text = f.read()
    # Skip occurrences of the header inside commented-out (old) versions of the table
    start = text.index(TABLE_HEADER)
    while text[text.rfind("\n", 0, start) + 1:start].lstrip().startswith("%"):
        start = text.index(TABLE_HEADER, start + 1)
    start += len(TABLE_HEADER)
    end = text.index("\\end{tabular}", start)
    body = table_body().replace(*FOOTNOTE_CELL)
    with open(path, "w") as f:
        f.write(text[:start] + body + text[end:])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--paper":
        update_paper(sys.argv[2])
    else:
        print(table_body(), end="")
