#!/bin/bash
#!/bin/sh
# Node resource configurations
#SBATCH --job-name=ava_train
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH -N 1
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

module load vector_cv_project

echo Installing SlowFast to your local space
pushd SlowFast
python setup.py build develop --prefix $HOME/.local/
popd

echo Running SlowFast ava example
cmd="python SlowFast/tools/run_net.py --cfg SlowFast/configs/AVA/SLOW_8x8_R50_SHORT.yaml NUM_GPUS 2 TRAIN.BATCH_SIZE 64 SOLVER.BASE_LR 0.0125 DATA_LOADER.ENABLE_MULTI_THREAD_DECODE True DATA_LOADER.NUM_WORKERS 4"

if [ -z "$SLURM_JOB_ID" ]
then
    echo ------------- FAILED ----------------
    echo \$SLURM_JOB_ID is empty, did you launch the script with slurm ?
    exit
else
    echo Running CMD: $cmd
fi


eval $cmd
