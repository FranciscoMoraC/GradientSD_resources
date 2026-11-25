# GradientSD: Gradient-Based Subgroup Discovery in High-Dimensional Data

## Algorithm and experiment code

The code for the GradientSD algorithm can be found in the `neuralsd` folder. `neuralsd/pruning.py` contains the post-hoc pruning step, which removes the selectors whose removal does not change the coverage of a subgroup on the discovery data.

The code for the experiments is split between each method applied.

We included reference for the data used in the main paper. All data but the Vancomycin dataset is available online. For the Vancomycin dataset, which was extracted from the MIMIC database, we reference the main work which provided the SQL queries.

The `data_generator` folder contains the code for generating the synthetic datasets used in the experiments, as well as the input file for generating the synthetic datasets with the desired structure (3 pure subgroups of varying number of selectors and samples).

The `Hyperparameter selection.pdf` file contains the hyperparameter grid and the final results for the hyperparameter selection process.

The `main.py` and `main_sdiga.py` files include code for running BSD and SDIGA and importing the data for both experiments.

The `nsdExperiments/gsd_cpu_time.py` script measures the runtime of GradientSD on CPU (single-threaded and multi-threaded, median of 5 runs) for the datasets of the comparison with BSD, and checks that the WRAcc obtained is the same as on GPU. Its results are in `nsdExperiments/gsd_cpu_times.csv`.

The `nsdExperiments/ablation_e1.py` script runs the ablation of Experiment 1: WRAcc of the best single selector (initial subgroup; `single_selector_wracc` on the discrete subgroup, as in Table 1 of the paper, and `initial_wracc` on the relaxed counts with the initial weights), relaxed and discrete WRAcc and epochs of GradientSD, and a sensitivity analysis over the learning rate (1, 10, 100), the patience (2, 10) and the initial weight magnitude (5, 10, 20). Results are in `nsdExperiments/ablation_e1.csv` and `nsdExperiments/ablation_e1_sensitivity.csv`.

## Post-hoc analysis

`posthoc_analysis.py` applies the pruning step of GradientSD to the stored bootstrap subgroups and re-evaluates them (pruned description length and in-bag/out-of-bag Youden's index), and computes the paired out-of-bag differences between GradientSD and every other method with 95% percentile intervals (`bootstrap_results/posthoc_*.csv`, `bootstrap_results/paired_differences.csv`). Some stored GradientSD results lack the subgroup descriptions; `rerun_descriptions.sh` re-runs them into `bootstrap_results_rerun/` (GradientSD is deterministic, so the re-runs reproduce the original subgroups), from which the descriptions are taken. `make_table2.py` builds Table 2 of the paper, with GradientSD reported after pruning (`python make_table2.py --paper <path to main.tex>` replaces it in place).

## ARCENE use case

The `useCaseArcene` folder contains the experiments on the ARCENE dataset:

- `data_preproc.ipynb`: discretization of the raw data into four quantile bins.
- `run.ipynb`: GradientSD applied to the whole dataset (subgroups reported in the paper).
- `convergence_plots.ipynb`: convergence and relative learning progress curves of the three subgroups over 10 bootstrap samples, omitted from the paper for space.
- `pca_beam_search.py`: grid search of the PCA + discretization + beam search baseline (results in `youden_results_pca_beam_search.csv`).
- `run_bootstrap_arcene.py`: out-of-bag evaluation of GradientSD and of the PCA + beam search baseline over 30 stratified bootstrap samples, with the same protocol as the high-dimensional experiment. Results are written to `bootstrap_results/GSD/arcene` and `bootstrap_results/PCA_Beam/arcene`, and summarized in `bootstrap_results/summary.csv` by `bootstrap_summary.py`. The single run and the bootstrap use `additional_params_for_weight_init={"n": 1}` (initial weight 0 for the best selector); on ARCENE it returns the same subgroups as the default initialization (weight 10).
- `run_extra_arcene.py beam`: beam search directly on the 26,357 selectors of the discretized dataset (ChiSquaredQF and StandardQF(0.5)), with the same bootstrap protocol (`bootstrap_results/Beam_chi2`, `bootstrap_results/Beam_std05`).
- `run_extra_arcene.py permutation`: permutation test of GradientSD (100 shuffles of the target; `bootstrap_results/GSD_permutation`).

## Other information

The `sdiga/sdiga.py` file contains the modified version of SDIGA used in the experiments. It is based on the implementation of the [subgroups](https://github.com/antoniolopezmc/subgroups) library (v0.1.11) and adds a maximum pattern size for the chromosomes of the initial population, a maximum number of subgroups to return, and a stopping condition when the best chromosome is empty, along with some performance improvements. Both `main_sdiga.py` and `sdiga/run_bootstrap.py` import this version instead of the one included in the library, so it runs on top of `subgroups==0.1.11` (the version listed in `requirements.txt` and used in the experiments).

A `requirements.txt` file is included with the used libraries and their versions. We recommend using a virtual environment to install the dependencies.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

