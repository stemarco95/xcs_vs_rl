#!/usr/bin/env bash
#SBATCH --job-name=deep_sarsa_frozenlake4x4_1
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/train_and_log/output_files/deep_sarsa/train_frozenlake4x4_1_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dir=$(pwd)

module load anaconda
conda activate research

srun python3 "$dir/../main.py" "train_and_log/sbatch/configs/deep_sarsa/frozenlake4x4_1.json"
