#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p p100
#SBATCH --cpus-per-task=2
#SBATCH --time=180:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=gabgab
#SBATCH --output=gabgab_job_%j.out

. /etc/profile.d/lmod.sh
. grandproj.env
module use /pkgs/environment-modules/
module load pytorch1.7.1-cuda10.2-python3.6
/h/skhalid/cv_vector/_runner.sh
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
#python /h/skhalid/pytorch.py
#wait
