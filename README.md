# Gradient Descent Subgroup Discovery without Dimensionality Reduction for Biomedical Data

## Algorithm and experiment code

The code for the NeuralSD algorithm can be found in the `neuralsd` folder.

The code for the experiments is split between each method applied.

We included reference for the data used in the main paper. All data but the Vancomycin dataset is available online. For the Vancomycin dataset, which was extracted from the MIMIC database, we reference the main work which provided the SQL queries.

The `data_generator` folder contains the code for generating the synthetic datasets used in the experiments, as well as the input file for generating the synthetic datasets with the desired structure (3 pure subgroups of varying number of selectors and samples).

The `Hyperparameter selection.pdf` file contains the hyperparameter grid and the final results for the hyperparameter selection process.

The `main.py` and `main_sdiga.py` files include code for running BSD and SDIGA and importing the data for both experiments.

The `sdiga/sdiga.py` file contains the modified version of SDIGA used in the experiments. It is based on the implementation of the [subgroups](https://github.com/antoniolopezmc/subgroups) library (v0.1.11) and adds a maximum pattern size for the chromosomes of the initial population, a maximum number of subgroups to return, and a stopping condition when the best chromosome is empty, along with some performance improvements. Both `main_sdiga.py` and `sdiga/run_bootstrap.py` import this version instead of the one included in the library, so it runs on top of `subgroups==0.1.11` (the version listed in `requirements.txt` and used in the experiments).

A `requierements.txt` file is included with the used libraries and their versions. We recommend using a virtual environment to install the dependencies.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

