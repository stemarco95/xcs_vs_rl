#!/usr/bin/env bash
#SBATCH --job-name=sarsa_taxi
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/optim/output_files/sarsa/optim_taxi_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
dir=$(pwd)

module load anaconda
conda activate research

srun python3 "$dir/../main.py" "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.5, 0.1, 0.05, 0.001]}, {\"name\": \"epsilon\", \"values\": [0.20, 0.25, 0.30, 0.35, 0.40]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}"
