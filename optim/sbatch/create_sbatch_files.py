import json
from pathlib import Path

from optim.sbatch.configs import CONFIGURATIONS


def create_sbatch(agent_name, agent_confs):
    file_names = []

    for conf in agent_confs:
        experiment = json.loads(conf)
        env = experiment['environment']['type']
        file_name = "optim_"
        file_name += env
        if env == 'frozenlake':
            if experiment['environment']['parameter']['desc_size'] == 4:
                file_name += "4x4"
            else:
                file_name += "8x8"

        print_file_name = file_name + "_%j.out"
        file_name += ".sl"
        file_names.append(file_name)

        filepath = Path(agent_name + "/" + file_name).absolute()
        conf = conf.replace('"', '\\"')
        with filepath.open("w", encoding="utf-8") as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(f"#SBATCH --job-name={agent_name}_{env}\n")
            f.write('#SBATCH --partition=epyc\n')
            f.write('#SBATCH --mem-per-cpu=1G\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --cpus-per-task=30\n')
            f.write("#SBATCH --time=7-0\n")
            f.write('#SBATCH --output=/hpc/gpfs2/home/u/stemarco/xcs-vs-mfrl/optim/output_files/' + agent_name + '/' + print_file_name + "\n\n")

            f.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')

            f.write("dir=$(pwd)\n\n")
            f.write("module load anaconda\n")
            f.write("conda activate research\n\n")

            f.write('srun python3 "$dir/../main.py" ' + "\"" + conf + "\"\n")

    return file_names


def create_bash_script(file_names, agent_name):
    filepath = Path(f"start_all_{agent_name}_sbatch.sh")

    with filepath.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n\n")
        for file_name in file_names:
            f.write("sbatch " + agent_name + "/" + file_name + "\n" + "sleep .5\n")


if __name__ == "__main__":
    for agent, confs in CONFIGURATIONS.items():
        files = create_sbatch(agent_confs=confs, agent_name=agent)
        create_bash_script(file_names=files, agent_name=agent)
