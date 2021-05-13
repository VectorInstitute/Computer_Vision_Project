#! /bin/bash
#SBATCH -p t4v2
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --job-name=semantic_unet_base
#SBATCH --output=semantic_unet_base.log
#SBATCH --ntasks=1

echo Running on $(hostname)
date
nvidia-smi
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
pwd

export PATH=/pkgs/anaconda3/bin:$PATH
which conda

#conda init bash
eval "$(conda shell.bash hook)"

conda activate <your own environment>

python train.py

conda deactivate