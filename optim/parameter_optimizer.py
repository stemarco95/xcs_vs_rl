import time
import optuna
import random
import threading
import numpy as np
import components.utils.json_handler as json_handler

from pathlib import Path
from components.utils import printer
from optim.runner.trial_runner import TrialRunner
from components.utils.utility_functions import set_seeds

lock = threading.Lock()


class ParameterOptimizer:
    def __init__(self, conf):
        set_seeds(conf['seed'])
        self.n_jobs = 30
        self.seeds = [random.randint(0, 2**32 - 1) for _ in range(7)]

        self.conf_env = conf['environment']
        self.conf_agent = conf['agent']
        self.h_params = conf['optim_parameter']

        self.process_id = -1
        self.trials = len(self.h_params) * 30

        self.json_folder_path = f"optim/output_files/temp/{self.conf_agent['type']}_{self.conf_env['type']}"

        if self.conf_env['type'] == 'frozenlake':
            if self.conf_env['parameter']['desc_size'] == 4:
                self.json_folder_path += "4x4"
            else:
                self.json_folder_path += "8x8"

        path = Path(self.json_folder_path)
        path.mkdir(parents=True, exist_ok=True)

        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=conf["seed"]))

    def get_process_name(self):
        process_name = self.conf_agent['type'] + "_" + self.conf_env['type']
        for param in self.conf_env['parameter']:
            process_name += "_" + str(self.conf_env['parameter'][param])

        return process_name

    def get_process_id(self):
        with lock:
            self.process_id += 1
            return self.process_id

    def run(self):
        self.study.optimize(self.objective, n_trials=self.trials, n_jobs=self.n_jobs)
        time.sleep(2)
        json_handler.print_from_folder(self.json_folder_path)
        json_handler.delete(self.json_folder_path)
        printer.print_best_param_and_value(self.study.best_params, self.study.best_value)
        return self.study.best_params, self.study.best_value

    def objective(self, trial):
        process_id = self.get_process_id()
        conf_file_path = self.json_folder_path + f"/{process_id}.json"
        json_handler.create_single_conf(conf_file_path)

        agent_dict = self.conf_agent.copy()
        for param in self.h_params:
            if 'values' in param:
                agent_dict['parameter'][param['name']] = trial.suggest_categorical(param['name'], param['values'])
            else:
                agent_dict['parameter'][param['name']] = trial.suggest_float(name=param['name'],
                                                                             low=param['range_min'],
                                                                             high=param['range_max'])

        overall_scores = []
        overall_iterations = []
        for i in range(0, 5):
            set_seeds(self.seeds[i])
            self.conf_env['parameter']['seed'] = self.seeds[i]
            runner = TrialRunner(self.conf_env, self.conf_agent, conf_file_path, process_id, i + 1)
            all_scores, iterations = runner.run()
            overall_scores.append(all_scores)
            overall_iterations.append(iterations)
            json_handler.save_result(conf_file_path, process_id, iterations, i+1)

        return self.get_objective_value(overall_scores, overall_iterations)

    @staticmethod
    def get_objective_value(overall_scores, overall_iterations):
        average_reward = np.mean([np.mean(scores) for scores in overall_scores])
        average_iterations = np.mean(overall_iterations)

        if average_iterations > 0:
            objective_value = average_reward / average_iterations
        else:
            objective_value = 0

        return objective_value
