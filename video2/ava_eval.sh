#!/bin/bash
#!/bin/sh
# Node resource configurations
#SBATCH --job-name=ava_eval
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --partition=t4v2
#SBATCH --gres=gpu:8
#SBATCH --qos=normal
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH -N 1
#SBATCH --open-mode=append


CKPT=${1:-"checkpoints/SLOWFAST_32x2_R101_50_50.pkl"}

echo `date`: Job $SLURM_JOB_ID is allocated resource

module load vector_cv_project

echo Installing SlowFast to your local space
pushd SlowFast
python setup.py build develop --prefix $HOME/.local/
popd

echo Running SlowFast ava example
cmd="python SlowFast/tools/run_net.py  --cfg SlowFast/configs/AVA/c2/SLOWFAST_32x2_R101_50_50.yaml  TEST.CHECKPOINT_FILE_PATH $CKPT TRAIN.ENABLE False  NUM_GPUS 8  DATA_LOADER.NUM_WORKERS 8 DATA_LOADER.PIN_MEMORY True  TEST.BATCH_SIZE 8"

if [ -z "$SLURM_JOB_ID" ]
then
    echo ------------- FAILED ----------------
    echo \$SLURM_JOB_ID is empty, did you launch the script with slurm ?
    exit
else
    echo Running CMD: $cmd
fi


eval $cmd
