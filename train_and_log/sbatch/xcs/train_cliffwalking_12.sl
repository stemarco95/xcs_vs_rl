#!/usr/bin/env bash
#SBATCH --job-name=xcs_cliffwalking_12
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/train_and_log/output_files/xcs/train_cliffwalking_12_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dir=$(pwd)

module load anaconda
conda activate research

srun python3 "$dir/../main.py" "train_and_log/sbatch/configs/xcs/cliffwalking_12.json"
