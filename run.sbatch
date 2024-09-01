#!/bin/sh
#SBATCH --job-name=RL-Project
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=120
#SBATCH --ntasks=1
#SBATCH --output=rl-project.log


# module load "architecture/AMD"

source env/bin/activate
python3 main.py --data-path data/$1.parquet --levels $2 --reward $3 --agent $4
deactivate


