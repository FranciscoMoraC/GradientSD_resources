#!/bin/bash
# Re-runs the bootstrap of the algorithms whose stored results lack the subgroup
# descriptions, writing to bootstrap_results_rerun/ so the original results (and their
# runtimes) are kept. The descriptions are only used by posthoc_analysis.py.
# SDIGA is not re-run (each iteration takes hours to days). Beam search on Facebook needs
# about 8 GB per process, hence the limited number of parallel iterations (BEAM_PARALLEL).
#
# Usage (from this folder):
#     nohup ./rerun_descriptions.sh [gpu|beam] > bootstrap_results_rerun/<log> 2>&1 &

PYTHON=${PYTHON:-python3}
DATASETS="synthetic polluted_synthetic Facebook brca_n"
export BOOTSTRAP_RESULTS_DIR=$(pwd)/bootstrap_results_rerun
mkdir -p $BOOTSTRAP_RESULTS_DIR

if [ "$1" == "gpu" ]; then
    export OMP_NUM_THREADS=4
    for folder in nsdExperiments diffnaps/code MCTS4DM; do
        for dataset in $DATASETS; do
            for iteration in $(seq 0 29); do
                (cd $folder && $PYTHON run_bootstrap.py $dataset $iteration) || echo "!!! $folder $dataset $iteration failed"
            done
        done
    done
elif [ "$1" == "beam" ]; then
    export OMP_NUM_THREADS=1
    for dataset in $DATASETS; do
        seq 0 29 | (cd beam && xargs -P ${BEAM_PARALLEL:-10} -I{} sh -c "$PYTHON run_bootstrap.py $dataset {} || echo '!!! beam $dataset {} failed'")
    done
fi
