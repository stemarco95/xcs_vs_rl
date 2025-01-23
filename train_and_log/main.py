import os
import sys
import json
import time
import pandas as pd

from pathlib import Path


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def get_experiment_name(conf) -> str:
    env_parameters = conf['environment']['parameter']

    relevant_params = {k: v for k, v in env_parameters.items() if
                       k not in ['desc_size', 'iterations', 'encoding_type', 'discretization_bins']}

    sorted_params = sorted(relevant_params.items())

    return "_".join(f"{k}_{v}" for k, v in sorted_params)


def run_experiment(conf, current_conf, num_confs):
    start_time = time.time()
    experiment_name = get_experiment_name(conf)

    set_seeds(conf['seed'])
    conf['environment']['parameter']['seed'] = conf['seed']

    runner = MainRunner(conf, current_conf, num_confs, eval_episodes=30)
    metrics = runner.run()

    execution_time = time.time() - start_time
    start_time = time.time()
    log_metrics_as_csv(conf, metrics, experiment_name)
    log_time = time.time() - start_time

    print(f"Execution time: {execution_time:.2f} seconds | Log time: {log_time:.2f} seconds")

    return sum(metrics['eval_succeeded_episodes'])


def log_metrics_as_csv(conf, metrics, experiment_name):
    env_type = conf['environment']['type']
    agent_type = conf['agent']['type']

    output_dir = Path('train_and_log', 'results', agent_type, env_type)
    if env_type == 'frozenlake':
        if conf['environment']['parameter']['desc_size'] == 4:
            output_dir = output_dir / '4x4'
        else:
            output_dir = output_dir / '8x8'

    output_dir = output_dir / experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)

    env_parameters = conf['environment']['parameter']
    params_env = {}
    for key, value in env_parameters.items():
        params_env[key] = value

    agent_parameters = conf['agent']['parameter']
    params_agent = {}
    for key, value in agent_parameters.items():
        if key != 'env':
            params_agent[key] = value

    params_env_path = output_dir / 'env_params.json'
    params_agent_path = output_dir / 'agent_params.json'

    with open(params_env_path, 'w') as f:
        json.dump(params_env, f)
    with open(params_agent_path, 'w') as f:
        json.dump(params_agent, f)

    output_dir = output_dir / str(conf['seed'])
    output_dir.mkdir(exist_ok=True, parents=True)

    results_path = output_dir / 'results.csv'
    df = pd.DataFrame(metrics)
    df.to_csv(results_path, index=False)


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working director:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()

    from components.utils.utility_functions import set_seeds
    from train_and_log.runner.main_runner import MainRunner

    file_path = sys.argv[1]
    confs = json.load(open(file_path))
    all_successful_episodes = 0
    all_episodes = 0
    for conf_num, conf in enumerate(confs):
        successful_episodes = run_experiment(conf, conf_num + 1, len(confs))
        all_successful_episodes += successful_episodes
        all_episodes += conf['environment']['parameter']['iterations'] * 30

    print(f"Successful episodes: {all_successful_episodes}/{all_episodes} ({all_successful_episodes / all_episodes * 100:.2f}%)")
    print("All Experiments done.")
