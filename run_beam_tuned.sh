#!/bin/bash
# Bootstrap evaluation of Beam Search with the hyperparameters (including the depth) selected
# on the discovery/validation split by beam/select_hyperparams_depth.py.
#
# Two chains: the small datasets start as soon as their hyperparameter search
# (beam/hyperparam_depth.log) finishes, and Facebook when its own search
# (beam/hyperparam_depth_facebook.log) finishes. Before selecting, the search is relaunched
# once to complete any failed run (finished runs are skipped).
#
# Usage (from this folder):
#     nohup ./run_beam_tuned.sh > bootstrap_results_tuned/run.log 2>&1 &

PYTHON=${PYTHON:-$HOME/miniconda3/envs/pytorch/bin/python}
ITERATIONS=30
export BOOTSTRAP_RESULTS_DIR=$(pwd)/bootstrap_results_tuned
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$BOOTSTRAP_RESULTS_DIR"

wait_for_search() {
    local log=$1 datasets=$2 parallel=$3
    until grep -q "all done" "$log" 2>/dev/null; do sleep 60; done
    echo "$(date) search finished for $datasets; completing failed runs"
    DATASETS="$datasets" PARALLEL=$parallel ./run_hyperparam_depth.sh >> "$log.retry" 2>&1
}

select_config() {
    local name=$1
    (cd beam && $PYTHON select_hyperparams_depth.py && cp selected_hyperparams_depth.csv "$BOOTSTRAP_RESULTS_DIR/selected_$name.csv")
}

run_dataset() {
    local dataset=$1 parallel=$2 config=$3
    echo "$(date) start $dataset"
    seq 0 $((ITERATIONS - 1)) | (cd beam && BEAM_CONFIG="$config" xargs -P "$parallel" -I{} \
        sh -c "$PYTHON run_bootstrap.py $dataset {} || echo '!!! beam $dataset {} failed'")
    echo "$(date) end $dataset"
}

(
    wait_for_search beam/hyperparam_depth.log "brca_n polluted_synthetic synthetic" 12
    select_config small
    for dataset in brca_n polluted_synthetic synthetic; do
        run_dataset $dataset 10 "$BOOTSTRAP_RESULTS_DIR/selected_small.csv"
    done
) > "$BOOTSTRAP_RESULTS_DIR/small.log" 2>&1 &

(
    wait_for_search beam/hyperparam_depth_facebook.log Facebook 12
    select_config facebook
    run_dataset Facebook 12 "$BOOTSTRAP_RESULTS_DIR/selected_facebook.csv"
) > "$BOOTSTRAP_RESULTS_DIR/facebook.log" 2>&1 &

wait
echo "$(date) all done"
