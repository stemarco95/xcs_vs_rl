import os
import sys


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

    from optim.parameter_optimizer import ParameterOptimizer
    from components.utils.utility_functions import handle_command_line
    from components.utils.json_handler import save_dict_as_json

    conf = handle_command_line(sys.argv[1])
    conf['environment']['parameter']['seed'] = conf['seed']
    runner = ParameterOptimizer(conf)
    best_params, best_value = runner.run()

    det_prob_action = conf['environment']['parameter'].get('det_prob_action', 1.0)
    det_prob_state = conf['environment']['parameter'].get('det_prob_state', 1.0)
    path = f"optim/results/{conf['agent']['type']}/{det_prob_action}_{det_prob_state}/{conf['environment']['type']}"

    output_dict = {
        "agent":
            {
                "type": conf['agent']['type'],
                "parameter": best_params,
            },
        "environment":
            {
                "type": conf['environment']['type'],
                "parameter": conf['environment']['parameter'],
            },
        "seed": conf['seed']
    }

    if conf['environment']['type'] == 'frozenlake':
        if conf['environment']['parameter']['desc_size'] == 4:
            path += "4x4"
        else:
            path += "8x8"
    elif conf['environment']['type'] == 'blackjack':
        path += "_" + str(conf['environment']['parameter']['natural'])

    save_dict_as_json(output_dict, path + ".json")
