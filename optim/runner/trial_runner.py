import numpy as np

from optim.runner.eval_runner import EvalRunner
from optim.runner.train_runner import TrainRunner
from components.utils.initializer import get_environment_instance, get_agent_instance
from components.utils.json_handler import save_optim_values, save_optim_run_info, save_run_number_of_trial


class TrialRunner:
    def __init__(self, conf_env, conf_agent, json_path, json_number, run_index):
        self.env = get_environment_instance(conf_env)
        self.agent = get_agent_instance(conf_agent, self.env)
        self.eval_runner = EvalRunner(self.env, self.agent)
        self.train_runner = TrainRunner(self.env, self.agent)
        self.json_path = json_path
        self.json_number = json_number
        self.run_index = run_index

        self.goal_threshold = self.env.get_goal_threshold()

        self.full_goal_threshold = self.goal_threshold * 0.95 if self.goal_threshold >= 0 else self.goal_threshold * 1.05
        self.num_eval_checks = 4
        self.avg_trains = []
        self.avg_evals = []

        if run_index == 1:
            save_optim_run_info(self.json_path, self.json_number, self.agent.get_agent_info(), self.env.get_env_info())
        save_run_number_of_trial(self.json_path, self.json_number, run_index)

    def run(self):
        loop_iteration = 0

        for loop_iteration in range(self.env.iterations):
            # Train
            self.avg_trains.append(np.mean(self.train_runner.run()))

            # Eval
            succeeded_episodes, score_list = self.eval_runner.run()
            self.avg_evals.append(np.mean(score_list))

            overall_mean_train_score = np.mean(self.avg_trains)
            overall_mean_eval_score = np.mean(self.avg_evals)

            if (loop_iteration + 1) % 50 == 0 or np.mean(score_list) >= self.goal_threshold:
                save_optim_values(self.json_path, loop_iteration + 1, self.env.iterations,
                                  self.avg_trains[-1], self.avg_evals[-1],
                                  overall_mean_train_score, overall_mean_eval_score, succeeded_episodes, self.run_index > 0)

            if np.mean(score_list) >= self.goal_threshold:
                check_list_score = [np.mean(score_list)]
                for _ in range(0, self.num_eval_checks):
                    succeeded_episodes, score_list = self.eval_runner.run()
                    check_list_score.append(np.mean(score_list))
                if np.mean(check_list_score) > self.full_goal_threshold:
                    self.avg_evals.extend(check_list_score)
                    return self.avg_evals, loop_iteration + 1

        return self.avg_evals, loop_iteration + 1
