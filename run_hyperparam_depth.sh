#!/bin/bash
# Beam Search hyperparameter search on the discovery split with depth in {3, 5}
# (see beam/run_hyperparam_depth.py). Longest configurations first; finished ones are skipped.
#
# Usage (from this folder):
#     DATASETS="brca_n polluted_synthetic synthetic" PARALLEL=12 nohup ./run_hyperparam_depth.sh > beam/hyperparam_depth.log 2>&1 &
PYTHON=${PYTHON:-$HOME/miniconda3/envs/pytorch/bin/python}
DATASETS=${DATASETS:-"brca_n polluted_synthetic synthetic Facebook"}
PARALLEL=${PARALLEL:-12}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd beam
for depth in 5 3; do for width in 250 100 25 10; do for dataset in $DATASETS; do
    for measure in chisquaredQF standardQF_0.5 standardQF_1.0; do
        echo "$dataset $measure $width $depth"
    done
done; done; done | xargs -P "$PARALLEL" -L1 sh -c "$PYTHON run_hyperparam_depth.py \$0 \$1 \$2 \$3 || echo \"!!! failed \$0 \$1 \$2 \$3\""
echo "$(date) all done"
