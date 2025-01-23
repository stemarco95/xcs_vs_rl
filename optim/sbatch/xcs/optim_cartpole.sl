#!/usr/bin/env bash
#SBATCH --job-name=xcs_cartpole
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=7-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/optim/output_files/xcs/optim_cartpole_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
dir=$(pwd)

module load anaconda
conda activate research

srun python3 "$dir/../main.py" "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.001, 0.005, 0.010, 0.015, 0.020]}, {\"name\": \"pop_size\", \"values\": [5000, 10000, 10000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 1000,  \"encoding_type\": \"gray_code\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}"
