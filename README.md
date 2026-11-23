# Gradient Descent Subgroup Discovery without Dimensionality Reduction for Biomedical Data

## Algorithm and experiment code

The code for the NeuralSD algorithm can be found in the `neuralsd` folder.

The code for the experiments is split between each method applied.

We included reference for the data used in the main paper. All data but the Vancomycin dataset is available online. For the Vancomycin dataset, which was extracted from the MIMIC database, we reference the main work which provided the SQL queries.

The `data_generator` folder contains the code for generating the synthetic datasets used in the experiments, as well as the input file for generating the synthetic datasets with the desired structure (3 pure subgroups of varying number of selectors and samples).

The `Hyperparameter selection.pdf` file contains the hyperparameter grid and the final results for the hyperparameter selection process.

The `main.py` and `main_sdiga.py` files include code for running BSD and SDIGA and importing the data for both experiments.

The `nsdExperiments/nsd_cpu_time.py` script measures the runtime of NeuralSD on CPU (single-threaded and multi-threaded, median of 5 runs) for the datasets of the comparison with BSD, and checks that the WRAcc obtained is the same as on GPU. Its results are in `nsdExperiments/nsd_cpu_times.csv`.

## ARCENE use case

The `useCaseArcene` folder contains the experiments on the ARCENE dataset:

- `data_preproc.ipynb`: discretization of the raw data into four quantile bins.
- `run.ipynb`: NeuralSD applied to the whole dataset (subgroups reported in the paper).
- `convergence_plots.ipynb`: convergence and relative learning progress curves of the three subgroups over 10 bootstrap samples, omitted from the paper for space.
- `pca_beam_search.py`: grid search of the PCA + discretization + beam search baseline (results in `youden_results_pca_beam_search.csv`).
- `run_bootstrap_arcene.py`: out-of-bag evaluation of NeuralSD and of the PCA + beam search baseline over 30 stratified bootstrap samples, with the same protocol as the high-dimensional experiment. Results are written to `bootstrap_results/NSD/arcene` and `bootstrap_results/PCA_Beam/arcene`, and summarized in `bootstrap_results/summary.csv` by `bootstrap_summary.py`.

## Other information

The `sdiga/sdiga.py` file contains the modified version of SDIGA used in the experiments. It is based on the implementation of the [subgroups](https://github.com/antoniolopezmc/subgroups) library (v0.1.11) and adds a maximum pattern size for the chromosomes of the initial population, a maximum number of subgroups to return, and a stopping condition when the best chromosome is empty, along with some performance improvements. Both `main_sdiga.py` and `sdiga/run_bootstrap.py` import this version instead of the one included in the library, so it runs on top of `subgroups==0.1.11` (the version listed in `requirements.txt` and used in the experiments).

A `requierements.txt` file is included with the used libraries and their versions. We recommend using a virtual environment to install the dependencies.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

