"""Figure of the scalability analysis of GradientSD (results of run_scalability.sh).

Four panels: runtime and peak GPU memory against the number of instances n (2000 noise
attributes, i.e. 10,012 selectors) and against the number of selectors, called p in the
paper (n = 7000; 5 selectors per noise attribute plus 12 informative selectors), as the
median over the repetitions with
the min-max range as error bars. Both axes are logarithmic and a dashed line of slope 1
(linear growth) through the first point is drawn as reference. Configurations that ran
out of GPU memory are marked on the x axis.

Usage (from the gsdExperiments folder):
    python plot_scalability.py [results_file] [output_pdf]
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

BASE_N, BASE_P = 7000, 2000
COLOR = "#2a6fb0"
X_TICKS = {"n": [3500, 7000, 14000, 28000], "selectors": [2500, 5000, 10000, 20000, 40000]}

results_file = sys.argv[1] if len(sys.argv) > 1 else "scalability_results.jsonl"
output_file = sys.argv[2] if len(sys.argv) > 2 else "scalability.pdf"
results = pd.read_json(results_file, lines=True)
results["gpu_peak_gb"] = results["gpu_peak_mb"] / 1024

# The base configuration (n = 7000, 2000 noise attributes) belongs to both axes; the x
# axis of the second pair of panels is the number of selectors (p in the paper)
axes_data = {
    "n": results[results["p"] == BASE_P],
    "selectors": results[results["n"] == BASE_N],
}
labels = {"n": "Instances $n$ ($p = 10{,}012$)", "selectors": "Selectors $p$ ($n = 7{,}000$)"}
measures = {"time_s": "Runtime (s)", "gpu_peak_gb": "Peak GPU memory (GB)"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.7), constrained_layout=True)
for ax, (variable, measure) in zip(axes, [(v, m) for v in axes_data for m in measures]):
    data = axes_data[variable]
    ok = data[data["status"] == "ok"]
    summary = ok.groupby(variable)[measure].agg(["median", "min", "max"]).reset_index()
    ax.errorbar(summary[variable], summary["median"],
                yerr=[summary["median"] - summary["min"], summary["max"] - summary["median"]],
                color=COLOR, marker="o", markersize=3.5, linewidth=1.5, capsize=2)
    # Linear growth reference through the first point
    x0, y0 = summary[variable].iloc[0], summary["median"].iloc[0]
    xs = summary[variable]
    ax.plot(xs, y0 * xs / x0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Plain numbers instead of powers of ten on both logarithmic axes
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(X_TICKS[variable]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    # First configuration that did not fit in GPU memory
    oom = data.loc[data["status"] == "oom", variable]
    if len(oom):
        ax.axvline(oom.min(), color="gray", linewidth=0.6, linestyle=":")
        ax.text(oom.min(), 0.04, "OOM ", rotation=90, va="bottom", ha="right", fontsize=6, color="gray",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel(labels[variable])
    ax.set_ylabel(measures[measure])
    ax.grid(True, which="major", color="0.9", linewidth=0.5)

fig.savefig(output_file)
print(f"Saved {output_file}")
print(results.groupby(["n", "p", "status"])[["time_s", "gpu_peak_mb", "rss_peak_mb", "youden"]].median())
