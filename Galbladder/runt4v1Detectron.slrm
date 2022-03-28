#!/bin/bash

# The lines that start with #SBATCH are read by slurm to set up the job
# any #SBATCH argument after the first non-empty/non-comment line will be ignored

#SBATCH --job-name=abc123
# Change this for a different type of GPU
#SBATCH --partition=t4v1

# Change this for a different quality of service (priority)
#SBATCH --qos=normal

# Change this for request different number of CPUs/GPU/Memory, they must fit on a single node
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# stdout/err are directed to file, these two arguments speficify where they should go, %j is a formatter for the job id
#SBATCH --output=./%j_testJob.out
#SBATCH --error=./%j_testJob.err

# Set the file mode to append, otherwise preemption resets the file and the previous output will be overwritten
#SBATCH --open-mode=append


if [ -z "$SLURM_JOB_ID" ]
then
    echo ------------- FAILED ----------------
    echo \$SLURM_JOB_ID is empty, did you launch the script with "sbatch" ?
    exit
else
    echo Job $SLRUM_JOB_ID is running
fi


module load vector_cv_project
hostname
which python
nvidia-smi

echo "This goes to stderr" 1>&2

wd=0.0001
ims=8
lr=0.00001
e=30
roi=512
#d='lr2e_5_ims_4_wd_1e_5'

touch $SLURM_JOB_ID'_'$wd'_'$ims'_'$lr'_'$e'_.txt'

python DetectronGBScript.py --wd $wd --ims $ims --lr $lr --e $e --roi $roi --d 'detectron2/output/'$wd'_'$ims'_'$lr'_'$e'/'

