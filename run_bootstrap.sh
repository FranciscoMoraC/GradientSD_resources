#!/bin/bash
# Bootstrap evaluation of every algorithm on every dataset.
#
# Each call to <algorithm folder>/run_bootstrap.py runs one iteration and writes
# bootstrap_results/<algorithm>/<dataset>/iteration_<i>.json. Iterations already stored
# are skipped, so this script can be relaunched after an interruption (or after an
# iteration is killed for running out of memory) to complete the missing ones.
#
# Two chains run at the same time, using at most 12 + 13 = 25 threads:
#   * NSD, DiffNaps and MCTS4DM run one iteration at a time with at most 12 threads,
#     and are compared by wall-clock time.
#   * Beam and SDIGA are single-threaded and much slower, so several iterations run in
#     parallel and they are compared by CPU time. SDIGA uses fewer parallel iterations
#     because it needs about 10-13 GB of memory per iteration on Facebook and brca_n.
#
# Usage (from this folder):
#     mkdir -p bootstrap_results
#     nohup ./run_bootstrap.sh > bootstrap_results/run.log 2>&1 &

PYTHON=${PYTHON:-python3}
ITERATIONS=30
DATASETS="synthetic polluted_synthetic Facebook brca_n"
BEAM_PARALLEL_ITERATIONS=13
SDIGA_PARALLEL_ITERATIONS=8

mkdir -p bootstrap_results

# Chain 1: one iteration at a time
(
    export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
    for folder in nsdExperiments diffnaps/code MCTS4DM; do
        for dataset in $DATASETS; do
            for iteration in $(seq 0 $((ITERATIONS - 1))); do
                (cd $folder && $PYTHON run_bootstrap.py $dataset $iteration) || echo "!!! $folder $dataset $iteration failed"
            done
        done
    done
) > bootstrap_results/run_wall_clock.log 2>&1 &

# Chain 2: several single-threaded iterations in parallel
(
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    for dataset in $DATASETS; do
        seq 0 $((ITERATIONS - 1)) | (cd beam && xargs -P $BEAM_PARALLEL_ITERATIONS -I{} \
            sh -c "$PYTHON run_bootstrap.py $dataset {} || echo '!!! beam $dataset {} failed'")
    done
    for dataset in $DATASETS; do
        seq 0 $((ITERATIONS - 1)) | (cd sdiga && xargs -P $SDIGA_PARALLEL_ITERATIONS -I{} \
            sh -c "$PYTHON run_bootstrap.py $dataset {} || echo '!!! sdiga $dataset {} failed'")
    done
) > bootstrap_results/run_cpu_time.log 2>&1 &

wait
$PYTHON bootstrap_summary.py
