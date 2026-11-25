#!/bin/bash
# Bootstrap evaluation of beam search with a larger maximum depth (default 5, the same
# maximum description length used for BSD in Experiment 1), keeping the hyperparameters
# selected with depth 3. Results go to a separate folder so the depth-3 results are kept.
#
# Facebook needs about 8 GB per process, so it runs with fewer parallel iterations, while
# the other datasets run one after another in a second chain. Iterations already stored
# are skipped, so the script can be relaunched after an interruption.
#
# Usage (from this folder):
#     nohup ./run_beam_depth.sh > bootstrap_results_depth5/run.log 2>&1 &

PYTHON=${PYTHON:-$HOME/miniconda3/envs/pytorch/bin/python}
ITERATIONS=30
FACEBOOK_PARALLEL=${FACEBOOK_PARALLEL:-10}
# About 3 GB per process on the synthetic datasets, while Facebook takes about 85 GB
OTHERS_PARALLEL=${OTHERS_PARALLEL:-8}

export BEAM_DEPTH=${BEAM_DEPTH:-5}
export BOOTSTRAP_RESULTS_DIR=${BOOTSTRAP_RESULTS_DIR:-$(pwd)/bootstrap_results_depth$BEAM_DEPTH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$BOOTSTRAP_RESULTS_DIR"

run_dataset() {
    local dataset=$1 parallel=$2
    echo "$(date) start $dataset"
    seq 0 $((ITERATIONS - 1)) | (cd beam && xargs -P "$parallel" -I{} \
        sh -c "$PYTHON run_bootstrap.py $dataset {} || echo '!!! beam $dataset {} failed'")
    echo "$(date) end $dataset"
}

run_dataset Facebook $FACEBOOK_PARALLEL > "$BOOTSTRAP_RESULTS_DIR/facebook.log" 2>&1 &
(
    for dataset in ${OTHER_DATASETS:-brca_n polluted_synthetic synthetic}; do
        run_dataset $dataset $OTHERS_PARALLEL
    done
) > "$BOOTSTRAP_RESULTS_DIR/others.log" 2>&1 &

wait
echo "$(date) all done"
