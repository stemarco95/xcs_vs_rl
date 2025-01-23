#!/usr/bin/env bash
#SBATCH --job-name=sarsa_blackjack
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/optim/output_files/sarsa/optim_blackjack_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
dir=$(pwd)

module purge
module load anaconda
conda activate research

srun python3 "$dir/../main.py" "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.20, 0.25, 0.30, 0.35, 0.40]}, {\"name\": \"gamma\", \"values\": [0.85, 0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}"
