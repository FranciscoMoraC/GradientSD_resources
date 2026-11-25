#!/bin/bash
# Grid of the scalability analysis of GradientSD (see scalability.py).
# Varies n with p = 2000 (as in Synthetic) and p with n = 7000, 3 repetitions each.
# Full-batch GradientSD needs about 14 times the size of the selector matrix in GPU
# memory, so on a 20 GB GPU the largest feasible configurations are n = 28,000
# (p = 2000) and p = 8,000 (n = 7000); the last value of each axis (n = 35,000 and
# p = 10,000) runs out of memory and marks that limit.
#
# Usage (from the nsd folder):
#     nohup bash run_scalability.sh > scalability.log 2>&1 &

PYTHON=~/miniconda3/envs/pytorch/bin/python
RESULTS=scalability_results.jsonl

for repetition in 0 1 2; do
    for n in 3500 7000 10500 14000 21000 28000 35000; do
        $PYTHON scalability.py $n 2000 $repetition $RESULTS
    done
    for p in 500 1000 3000 4000 6000 8000 10000; do
        $PYTHON scalability.py 7000 $p $repetition $RESULTS
    done
done
echo "DONE"
