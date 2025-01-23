#!/usr/bin/env bash
#SBATCH --job-name=dqn_frozenlake
#SBATCH --partition=epyc
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=7-0
#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/optim/output_files/dqn/optim_frozenlake8x8_%j.out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
dir=$(pwd)

module purge
module load anaconda
conda activate research

python3 "$dir/../main.py" "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]}, {\"name\": \"epsilon\", \"values\": [0.20, 0.25, 0.30, 0.35, 0.40]}, {\"name\": \"gamma\", \"values\": [0.85, 0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"binary\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}"
