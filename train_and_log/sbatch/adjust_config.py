import json


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def modify_json(data):
    for agent_name, agent_configs in data.items():
        for config in agent_configs:
            env = config['environment']
            params = env['parameter']
            if env['type'] != 'blackjack':
                if params['det_prob_action'] == 1.0 and params['det_prob_state'] == 1.0:
                    params['iterations'] = 400
                elif params['det_prob_action'] == 1.0 and params['det_prob_state'] == 0.95:
                    params['iterations'] = 750
                elif params['det_prob_action'] == 1.0 and params['det_prob_state'] == 0.90:
                    params['iterations'] = 1500
                elif params['det_prob_action'] == 0.95 and params['det_prob_state'] == 1.00:
                    params['iterations'] = 750
                elif params['det_prob_action'] == 0.90 and params['det_prob_state'] == 1.00:
                    params['iterations'] = 1500
                elif params['det_prob_action'] == 0.95 and params['det_prob_state'] == 0.95:
                    params['iterations'] = 1500
                elif params['det_prob_action'] == 0.95 and params['det_prob_state'] == 0.90:
                    params['iterations'] = 3000
                elif params['det_prob_action'] == 0.90 and params['det_prob_state'] == 0.95:
                    params['iterations'] = 3000
                elif params['det_prob_action'] == 0.90 and params['det_prob_state'] == 0.90:
                    params['iterations'] = 5000
            else:
                if params['det_prob_state'] == 1.0:
                    params['iterations'] = 400
                if params['det_prob_state'] == 0.95:
                    params['iterations'] = 750
                if params['det_prob_state'] == 0.90:
                    params['iterations'] = 1500

    return data


def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    filename = 'all_configs.json'

    data = load_json(filename)
    data = modify_json(data)
    save_json(data, filename)
    print("JSON successfully edited and saved.")


if __name__ == "__main__":
    main()
