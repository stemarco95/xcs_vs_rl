import random
import numpy as np

from components.utils.observation import Observation
from components.environments.abstract_env import AbstractEnv
from gymnasium.envs.toy_text import CliffWalkingEnv as GymCliffWalkingEnv
from components.utils.utility_functions import get_one_hot_state, get_binary_state, get_gray_code_state


# https://gymnasium.farama.org/environments/toy_text/cliff_walking/
class CliffWalkingEnv(GymCliffWalkingEnv, AbstractEnv):
    def __init__(self, render_mode=None, iterations=25, train_episodes=30, det_prob_state=1.0, det_prob_action=1.0,
                 encoding_type='decimal', seed=21):
        GymCliffWalkingEnv.__init__(self, render_mode=render_mode)
        AbstractEnv.__init__(self, iterations=iterations, train_episodes=train_episodes, max_payoff=0.0)

        self.det_prob_state = det_prob_state
        self.det_prob_action = det_prob_action
        self.encoding_type = encoding_type
        self.set_seed(seed)
        self.step_count = 0

        self.bad_state_indices = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        self.orthogonal_actions = {0: [1, 3],
                                   1: [0, 2],
                                   2: [1, 3],
                                   3: [0, 2]}

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        super().reset(seed=seed)

    def get_env_info(self):
        return {"Env name": "Cliff Walking Environment",
                "Number of iterations": self.iterations,
                "Number of training episodes per iteration": self.train_episodes,
                "Deterministic action probability": self.det_prob_action,
                "Deterministic state probability": self.det_prob_state}

    def get_state_dim(self):
        if self.encoding_type == 'one_hot':
            return 48
        if self.encoding_type == 'gray_code' or self.encoding_type == 'binary':
            return 6

        return 1  # decimal

    def get_action_dim(self):
        return 4

    def get_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        action = self.choose_action_with_probability(action)
        obs = Observation(*super().step(action))
        obs.state = self.make_noisy_state_if_necessary(obs.state)
        obs.state = self.encode_state(obs.state)
        obs.info['success'] = True if obs.reward == -1 and obs.terminated else False

        self.step_count += 1
        if self.step_count == 200:
            obs.terminated = True

        return obs

    def reset(self, **kwargs):
        self.step_count = 0
        state, info = super().reset()

        return Observation(self.encode_state(state), None, None, None, info)

    def get_goal_threshold(self):
        return -17

    def encode_state(self, state):
        if self.encoding_type == 'one_hot':
            return get_one_hot_state(state, self.get_state_dim())
        if self.encoding_type == 'binary':
            return get_binary_state(state, self.get_state_dim())
        if self.encoding_type == 'gray_code':
            return get_gray_code_state(state, self.get_state_dim())

        return state  # decimal

    def choose_action_with_probability(self, current_action):
        if random.random() <= self.det_prob_action:
            return current_action
        else:
            return random.choice(self.orthogonal_actions[current_action])

    def make_noisy_state_if_necessary(self, state):
        if random.random() > self.det_prob_state:
            row, col = self.decode(state)
            direction = random.choice(['up', 'down', 'left', 'right'])  # Noise on the x- and y-axis

            if direction == 'up' and row > 0:
                row -= 1
            elif direction == 'down' and row < 4 - 1:  # 3 is maximum
                row += 1
            elif direction == 'left' and col > 0:
                col -= 1
            elif direction == 'right' and col < 12 - 1:  # 11 is maximum
                col += 1
            new_state = self.encode(row, col)

            state = state if new_state in self.bad_state_indices else new_state
        return state

    @staticmethod
    def decode(state):
        row = state // 12
        col = state % 12
        return row, col

    @staticmethod
    def encode(row, col):
        return row * 3 + col


if __name__ == "__main__":
    env = CliffWalkingEnv(render_mode='human')
    observation = env.reset()

    reward = 0
    rewards = []
    eval_runs = 30
    successful_episodes = 0
    min_successful_eval_episodes = 100000
    max_successful_eval_episodes = -100000

    for _ in range(5000):
        successful_episodes_eval_run = 0
        for _ in range(eval_runs):
            while True:
                observation = env.step(env.action_space.sample())
                reward += observation.reward

                if observation.terminated or observation.truncated:
                    successful_episodes_eval_run += observation.info['success']
                    observation = env.reset()
                    rewards.append(reward)
                    reward = 0
                    break

        successful_episodes += successful_episodes_eval_run
        if successful_episodes_eval_run > max_successful_eval_episodes:
            max_successful_eval_episodes = successful_episodes_eval_run
        elif successful_episodes_eval_run < min_successful_eval_episodes:
            min_successful_eval_episodes = successful_episodes_eval_run

    max_successful_eval_episodes_percentage = (max_successful_eval_episodes/eval_runs) * 100
    min_successful_eval_episodes_percentage = (min_successful_eval_episodes/eval_runs) * 100
    print(f"Successful episodes: {successful_episodes}")
    print(f"Max successful episode in one eval run: {max_successful_eval_episodes} ({max_successful_eval_episodes_percentage}%)")
    print(f"Min successful episode in one eval run: {min_successful_eval_episodes} ({min_successful_eval_episodes_percentage}%)")
    print(f"Mean reward: {np.mean(rewards)}")
    env.close()
