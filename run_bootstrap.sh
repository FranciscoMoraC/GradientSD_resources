#!/bin/bash
# Bootstrap evaluation (30 iterations) of every algorithm on the four high-dimensional datasets.
# Iterations already stored in bootstrap_results/ are skipped.
# Beam and SDIGA are single-threaded, so we ran several of their iterations in parallel.

DATASETS="synthetic polluted_synthetic Facebook brca_n"

for folder in gsdExperiments diffnaps/code MCTS4DM beam sdiga; do
    for dataset in $DATASETS; do
        for iteration in $(seq 0 29); do
            (cd $folder && python3 run_bootstrap.py $dataset $iteration)
        done
    done
done

python3 bootstrap_summary.py
python3 pruning_gsd.py
