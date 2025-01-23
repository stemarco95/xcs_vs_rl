import json
import copy
import os
import random


def generate_seeds():
    random.seed(21)
    return [random.randint(0, 1000000) for _ in range(30)]


def get_config(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def get_iterations(env_type, det_probs):
    if env_type != 'blackjack':
        params_dict = {
            "1.0_1.0": 400,
            "1.0_0.95": 750,
            "1.0_0.9": 1500,
            "0.95_1.0": 750,
            "0.9_1.0": 1500,
            "0.95_0.95": 1500,
            "0.95_0.9": 3000,
            "0.9_0.95": 3000,
            "0.9_0.9": 5000
        }
    else:
        params_dict = {
            "1.0_True": 400,
            "0.95_True": 750,
            "0.9_True": 1500,
            "1.0_False": 400,
            "0.95_False": 750,
            "0.9_False": 1500
        }

    return params_dict[det_probs]


def get_configs(results_dir_path):
    configs = {}
    seeds = generate_seeds()

    agent_dirs = os.listdir(results_dir_path)
    for agent_dir in agent_dirs:
        agent_dir_path = os.path.join(results_dir_path, agent_dir)
        if os.path.isdir(agent_dir_path):
            configs[agent_dir] = {}

            det_prob_dirs = os.listdir(agent_dir_path)
            for det_prob_dir in det_prob_dirs:
                det_prob_dir_path = os.path.join(agent_dir_path, det_prob_dir)
                if os.path.isdir(det_prob_dir_path):

                    config_files = os.listdir(det_prob_dir_path)
                    for config_file in config_files:
                        if config_file.endswith('.json'):
                            config_file_path = os.path.join(det_prob_dir_path, config_file)
                            current_config = get_config(config_file_path)
                            env_type = current_config['environment']['type']

                            det_prob_action = current_config['environment']['parameter'].get('det_prob_action', 1.0)
                            det_prob_state = current_config['environment']['parameter'].get('det_prob_state', 1.0)

                            if env_type == 'blackjack':
                                natural = current_config['environment']['parameter']['natural']
                                det_probs = f"{det_prob_state}_{natural}"
                            else:
                                det_probs = f"{det_prob_action}_{det_prob_state}"

                            iterations = get_iterations(env_type, det_probs)
                            current_config['environment']['parameter']['iterations'] = iterations
                            if env_type == 'frozenlake':
                                desc_size = current_config['environment']['parameter']['desc_size']
                                env_type = f"{env_type}{desc_size}x{desc_size}"

                            if env_type not in configs[agent_dir]:
                                configs[agent_dir][env_type] = {seed: [] for seed in seeds}

                            for seed in seeds:
                                seeded_config = copy.deepcopy(current_config)
                                seeded_config['seed'] = seed
                                configs[agent_dir][env_type][seed].append(seeded_config)

    return configs


if __name__ == "__main__":
    derive_parameters = False

    results_dir = "../../optim/results"
    configs = get_configs(results_dir)

    for agent, envs in configs.items():
        for env, seeds in envs.items():
            if env == 'blackjack':
                all_configs = []
                for seed, configs_list in seeds.items():
                    all_configs.extend(configs_list)
                write_json(f'configs/{agent}/{env}.json', all_configs)

            elif env == 'frozenlake4x4':
                jsons = [[] for _ in range(2)]
                for index, (seed, configs_list) in enumerate(seeds.items()):
                    jsons[index % 2].extend(configs_list)
                for index, env_json in enumerate(jsons):
                    write_json(f'configs/{agent}/{env}_{index}.json', env_json)

            elif env == 'cartpole':
                for seed_index, (seed, configs_list) in enumerate(seeds.items(), start=1):
                    sorted_configs = sorted(configs_list, key=lambda x: x['environment']['parameter']['iterations'])
                    first_group = sorted_configs[:6]
                    second_group = sorted_configs[6:]

                    write_json(f'configs/{agent}/{env}_{seed_index}_1.json', first_group)
                    write_json(f'configs/{agent}/{env}_{seed_index}_2.json', second_group)

            else:
                for seed_index, (seed, configs_list) in enumerate(seeds.items(), start=1):
                    write_json(f'configs/{agent}/{env}_{seed_index}.json', configs_list)
