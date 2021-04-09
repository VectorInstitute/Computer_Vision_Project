#!/bin/bash

module load vector_cv_project

echo Installing SlowFast to your local space
pushd SlowFast
python setup.py build develop --prefix $HOME/.local/
popd

echo Running SlowFast ava example
cmd="python SlowFast/tools/run_net.py --cfg SlowFast/configs/AVA/SLOW_8x8_R50_SHORT.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 "

if [ -z "$SLURM_JOB_ID" ]
then
    echo ------------- FAILED ----------------
    echo \$SLURM_JOB_ID is empty, did you launch the script with slurm ?
    exit
else
    echo Running CMD: $cmd
fi


eval $cmd
