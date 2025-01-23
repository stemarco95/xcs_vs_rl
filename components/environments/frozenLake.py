import random
import numpy as np

from components.utils.observation import Observation
from components.environments.abstract_env import AbstractEnv
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from components.utils.utility_functions import get_one_hot_state, get_binary_state, get_gray_code_state


class FrozenLakeEnv(GymFrozenLakeEnv, AbstractEnv):
    def __init__(self, render_mode=None, desc_size=4, iterations=25, train_episodes=30, det_prob_state=1.0,
                 det_prob_action=1.0, encoding_type='decimal', seed=21):

        self.desc_size = desc_size
        GymFrozenLakeEnv.__init__(self, is_slippery=False, render_mode=render_mode, desc=self.get_desc())
        AbstractEnv.__init__(self, iterations=iterations, train_episodes=train_episodes, max_payoff=1.0)

        self.det_prob_state = det_prob_state
        self.det_prob_action = det_prob_action
        self.encoding_type = encoding_type
        self.set_seed(seed)
        self.step_count = 0

        if desc_size == 4:
            self.bad_state_indices = [5, 7, 11, 12]
        else:
            self.bad_state_indices = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59]

        self.orthogonal_actions = {
            0: [1, 3],  # Orthogonal actions for "Move left"
            1: [0, 2],  # Orthogonal actions for "Move down"
            2: [1, 3],  # Orthogonal actions for "Move right"
            3: [0, 2]  # Orthogonal actions for "Move up"
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        super().reset(seed=seed)

    def get_env_info(self):
        return {"Env name": "Frozen Lake Environment",
                "Number of iterations": self.iterations,
                "Number of training episodes per iteration": self.train_episodes,
                "Deterministic action probability": self.det_prob_action,
                "Deterministic state probability": self.det_prob_state}

    def get_state_dim(self):
        if self.encoding_type == 'one_hot':
            return self.desc_size * self.desc_size
        if self.encoding_type == 'gray_code' or self.encoding_type == 'binary':
            if self.desc_size == 4:
                return 4
            if self.desc_size == 8:
                return 6

        return 1  # decimal

    def get_action_dim(self):
        return 4

    def get_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        action = self.choose_action_with_probability(action)
        obs = Observation(*super().step(action))
        obs.state = self.get_boundary_state_if_necessary(obs.state)
        obs.state = self.encode_state(obs.state)
        obs.info['success'] = True if obs.reward == 1 else False

        self.step_count += 1
        if self.desc_size == 4 and self.step_count == 100:
            obs.terminated = True
        if self.desc_size == 8 and self.step_count == 200:
            obs.terminated = True

        return obs

    def reset(self, **kwargs):
        self.step_count = 0
        state, info = super().reset()
        state = self.encode_state(state)
        return Observation(state, None, False, False, info)

    def get_goal_threshold(self):
        return 1

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

    def get_desc(self):
        if self.desc_size == 4:
            return ["SFFF",
                    "FHFH",
                    "FFFH",
                    "HFFG"]
        else:
            return ["SFFFFFFF",
                    "FFFFFFFF",
                    "FFFHFFFF",
                    "FFFFFHFF",
                    "FFFHFFFF",
                    "FHHFFFHF",
                    "FHFFHFHF",
                    "FFFHFFFG"]

    def get_boundary_state_if_necessary(self, state):
        if random.random() > self.det_prob_state:
            row, col = self.decode(state)
            direction = random.choice(['up', 'down', 'left', 'right'])  # Noise on the x- and y-axis

            if direction == 'up' and row > 0:
                row -= 1
            elif direction == 'down' and row < self.desc_size - 1:
                row += 1
            elif direction == 'left' and col > 0:
                col -= 1
            elif direction == 'right' and col < self.desc_size - 1:
                col += 1
            new_state = self.encode(row, col)

            return state if new_state in self.bad_state_indices else new_state
        return state

    def decode(self, state):
        row = state // self.desc_size
        col = state % self.desc_size
        return row, col

    def encode(self, row, col):
        return row * self.desc_size + col


if __name__ == "__main__":
    DESC_SIZE = 4
    env = FrozenLakeEnv(encoding_type='decimal', desc_size=DESC_SIZE, det_prob_action=1.00, det_prob_state=1.00)
    observation = env.reset()

    if DESC_SIZE == 4:
        best_actions = {
            0: 2, 1: 2, 2: 1, 3: 0,
            4: 1, 5: 1, 6: 1, 7: 2,
            8: 2, 9: 1, 10: 1, 11: 1,
            12: 1, 13: 2, 14: 2, 15: 1
        }
    else:
        best_actions = {
            0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1,
            8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1,
            16: 2, 17: 2, 18: 3, 19: 2, 20: 2, 21: 2, 22: 2, 23: 1,
            24: 2, 25: 2, 26: 2, 27: 2, 28: 3, 29: 2, 30: 2, 31: 1,
            32: 2, 33: 2, 34: 3, 35: 2, 36: 2, 37: 2, 38: 2, 39: 1,
            40: 3, 41: 2, 42: 2, 43: 2, 44: 3, 45: 3, 46: 2, 47: 1,
            48: 3, 49: 2, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 55: 1,
            56: 2, 57: 2, 58: 3, 59: 2, 60: 2, 61: 2, 62: 2, 63: 1
        }

    reward = 0
    rewards = []
    eval_runs = 30
    successful_episodes = 0
    min_successful_eval_episodes = 100000
    max_successful_eval_episodes = -100000

    for _ in range(50000):
        successful_episodes_eval_run = 0
        for _ in range(eval_runs):
            while True:
                observation = env.step(
                    best_actions[observation.state])  # TODO replace with best policy
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

    max_successful_eval_episodes_percentage = (max_successful_eval_episodes / eval_runs) * 100
    min_successful_eval_episodes_percentage = (min_successful_eval_episodes / eval_runs) * 100
    print(f"Successful episodes: {successful_episodes}")
    print(
        f"Max successful episode in one eval run: {max_successful_eval_episodes} ({max_successful_eval_episodes_percentage}%)")
    print(
        f"Min successful episode in one eval run: {min_successful_eval_episodes} ({min_successful_eval_episodes_percentage}%)")
    print(f"Mean reward: {np.mean(rewards)}")
    env.close()
