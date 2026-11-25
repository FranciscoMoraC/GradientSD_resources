# GradientSD: Gradient-Based Subgroup Discovery in High-Dimensional Data

## Algorithm

The code for the GradientSD algorithm can be found in the `neuralsd` folder (`pruning.py` contains the post-hoc pruning step).

## Data

We included reference for the data used in the main paper. All data but the Vancomycin dataset is available online. For the Vancomycin dataset, which was extracted from the MIMIC database, we reference the main work which provided the SQL queries.

The `data_generator` folder contains the code for generating the synthetic datasets used in the experiments, as well as the input file for generating the synthetic datasets with the desired structure (3 pure subgroups of varying number of selectors and samples).

## Experiments

The code for the experiments is split between each method applied. The `main.py` and `main_sdiga.py` files include code for running BSD and SDIGA and importing the data for both experiments.

- Experiment 1 (comparison with BSD): `nsdExperiments/e1.ipynb`, `nsdExperiments/run_bsd.sh` and `nsdExperiments/bsd_results`. `nsdExperiments/ablation_e1.py` computes the WRAcc of the best single selector and the sensitivity analysis, and `nsdExperiments/gsd_cpu_time.py` the runtimes of GradientSD on CPU.
- Experiment 2 (high-dimensional datasets): the hyperparameter grids and the selected configurations are in `Hyperparameter selection.pdf`. Each method has a `run_bootstrap.py` script that runs one bootstrap iteration, and `run_bootstrap.sh` runs all of them. Results are stored in `bootstrap_results`, `bootstrap_summary.py` summarizes them and `pruning_gsd.py` applies the pruning step to the subgroups of GradientSD and computes the paired differences between GradientSD and the rest of the methods. Some results of GradientSD were stored without the subgroup descriptions, so we re-ran those iterations (`bootstrap_results_rerun`, with `BOOTSTRAP_RESULTS_DIR=../bootstrap_results_rerun python run_bootstrap.py <dataset> <iteration>`). `bootstrap_results/Beam_depth3` contains beam search on Facebook with the selected configuration but depth 3, used to compare the runtime with depth 5.
- Scalability (Fig. 1): `nsdExperiments/scalability.py` runs GradientSD on one dataset generated with the structure of Synthetic and records its runtime and peak GPU memory, `nsdExperiments/run_scalability.sh` runs the whole grid (results in `nsdExperiments/scalability_results.jsonl`) and `nsdExperiments/plot_scalability.py` draws the figure.
- ARCENE (`useCaseArcene`): `data_preproc.ipynb` discretizes the data, `run.ipynb` runs GradientSD on the whole dataset, `pca_beam_search.py` is the grid search of the PCA baseline, `run_bootstrap_arcene.py` the bootstrap evaluation of GradientSD and the PCA baseline, and `run_extra_arcene.py` the beam search on the original selectors and the permutation test.

## Other information

The `sdiga/sdiga.py` file contains the modified version of SDIGA used in the experiments. It is based on the implementation of the [subgroups](https://github.com/antoniolopezmc/subgroups) library (v0.1.11) and adds a maximum pattern size for the chromosomes of the initial population, a maximum number of subgroups to return and a stopping condition when the best chromosome is empty.

A `requirements.txt` file is included with the used libraries and their versions. We recommend using a virtual environment to install the dependencies.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
