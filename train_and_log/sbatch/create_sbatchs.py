import re

from pathlib import Path


def extract_key(path):
    parts = re.split(r'(\d+)', path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def create_sbatch(agent):
    file_names = []

    config_dir = Path(f'configs/{agent}')
    for json_path in sorted(config_dir.glob('*.json'), key=extract_key):
        env = json_path.stem
        file_name = f"train_{env}.sl"

        print_file_name = f"train_{env}_%j.out"
        file_names.append(file_name)

        filepath = Path(agent + "/" + file_name).absolute()
        with filepath.open("w", encoding="utf-8") as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(f"#SBATCH --job-name={agent}_{env}\n")
            f.write('#SBATCH --partition=epyc\n')
            f.write('#SBATCH --mem-per-cpu=1G\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --cpus-per-task=1\n')
            f.write("#SBATCH --time=7-0\n")
            f.write('#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/train_and_log/output_files/' + agent + '/' + print_file_name + "\n\n")
            f.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n\n')
            f.write("dir=$(pwd)\n\n")
            f.write("module load anaconda\n")
            f.write("conda activate research\n\n")
            f.write(f'srun python3 "$dir/../main.py" "train_and_log/sbatch/{json_path}"\n')

    return file_names


def create_bash_script(file_names, agent):
    filepath = Path(f"start_all_{agent}_sbatch.sh")

    with filepath.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n\n")
        for file_name in file_names:
            f.write("sbatch " + agent + "/" + file_name + "\n" + "sleep .5\n")


if __name__ == "__main__":
    agents = ["dqn", "deep_sarsa", "q_learning", "sarsa", "xcs"]
    for agent in agents:
        files = create_sbatch(agent)
        create_bash_script(files, agent)
