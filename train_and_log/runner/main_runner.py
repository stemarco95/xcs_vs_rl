import datetime
import numpy as np

from components.utils.initializer import get_instances
from train_and_log.runner.eval_runner import EvalRunner
from train_and_log.runner.train_runner import TrainRunner
from components.utils.printer import print_train_eval_values, print_train_run_info


class MainRunner:
    def __init__(self, conf, current_conf, num_confs, eval_episodes=30):
        self.conf = conf

        self.env, self.agent = get_instances(conf['environment'], conf['agent'])
        self.eval_episodes = eval_episodes
        self.training_iterations = self.env.iterations

        self.eval_runner = EvalRunner(self.env, self.agent, self.eval_episodes)
        self.train_runner = TrainRunner(self.env, self.agent)

        det_prob_state = getattr(self.env, 'det_prob_state', 1.0)
        det_prob_action = getattr(self.env, 'det_prob_action', 1.0)

        det_prob_complete = 1 - det_prob_state + 1 - det_prob_action
        goal_threshold = self.env.get_goal_threshold()
        self.full_goal_threshold = goal_threshold * (1 - det_prob_complete/2) if goal_threshold >= 0 else goal_threshold * (1 + det_prob_complete/2)

        self.num_eval_checks = 4
        print_train_run_info(self.agent.get_agent_info(), self.env.get_env_info(), current_conf, num_confs)

    def run(self):
        metrics = {
            "eval_scores": [],
            "eval_steps_per_episode": [],
            "eval_succeeded_episodes": []
        }

        all_eval_scores = []
        train = True
        for loop_iteration in range(self.training_iterations):
            if train:
                _ = self.train_runner.run()

            # Eval
            start = datetime.datetime.now()
            eval_scores, eval_steps_per_episode, succeeded_eval_episodes = self.eval_runner.run()
            avg_time_eval = (datetime.datetime.now() - start).total_seconds() / self.env.train_episodes

            all_eval_scores += eval_scores

            if np.mean(eval_scores) >= self.full_goal_threshold:
                check_list_score = [np.mean(eval_scores)]
                for _ in range(0, self.num_eval_checks):
                    score_list, _, _ = self.eval_runner.run()
                    check_list_score.append(np.mean(score_list))
                if np.mean(check_list_score) >= self.full_goal_threshold:
                    train = False

            if (loop_iteration + 1) % 25 == 0:
                print_train_eval_values(loop_iteration + 1, self.training_iterations, np.mean(eval_scores), np.mean(all_eval_scores), avg_time_eval, sum(succeeded_eval_episodes), train)

            metrics["eval_scores"].extend(eval_scores)
            metrics["eval_succeeded_episodes"].extend(succeeded_eval_episodes)
            metrics["eval_steps_per_episode"].extend(eval_steps_per_episode)

        return metrics
