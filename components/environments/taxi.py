import random
import numpy as np

from components.utils.observation import Observation
from components.environments.abstract_env import AbstractEnv
from gymnasium.envs.toy_text.taxi import TaxiEnv as GymTaxiEnv
from components.utils.utility_functions import get_one_hot_state, get_binary_state, get_gray_code_state


class TaxiEnv(GymTaxiEnv, AbstractEnv):
    def __init__(self, render_mode=None, iterations=25, train_episodes=30, det_prob_state=1.0, det_prob_action=1.0,
                 encoding_type='decimal', seed=21):
        GymTaxiEnv.__init__(self, render_mode=render_mode)
        AbstractEnv.__init__(self, iterations=iterations, train_episodes=train_episodes, max_payoff=20.0)

        self.det_prob_state = det_prob_state
        self.det_prob_action = det_prob_action
        self.encoding_type = encoding_type
        self.set_seed(seed)
        self.step_count = 0

        self.orthogonal_actions = {
            0: [2, 3],  # Orthogonal actions for "Move down"
            1: [2, 3],  # Orthogonal actions for "Move up"
            2: [0, 1],  # Orthogonal actions for "Move right"
            3: [0, 1],  # Orthogonal actions for "Move left"
            4: [4],  # Do not change pickup action
            5: [5]  # Do not change drop off action
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        super().reset(seed=seed)

    def get_env_info(self):
        return {"Env name": "Taxi Environment",
                "Number of iterations": self.iterations,
                "Number of training episodes per iteration": self.train_episodes,
                "Deterministic action probability": self.det_prob_action,
                "Deterministic state probability": self.det_prob_state}

    def get_state_dim(self):
        if self.encoding_type == 'one_hot':
            return 500
        if self.encoding_type == 'gray_code' or self.encoding_type == 'binary':
            return 9

        return 1  # decimal

    def get_action_dim(self):
        return 6

    def get_actions(self):
        return [0, 1, 2, 3, 4, 5]

    def step(self, action):
        action = self.choose_action_with_probability(action)
        obs = Observation(*super().step(action))
        obs.state = self.make_noisy_state_if_necessary(obs.state)
        obs.state = self.encode_state(obs.state)
        obs.info['success'] = True if obs.terminated else False

        self.step_count += 1
        if self.step_count == 200:
            obs.terminated = True

        return obs

    def reset(self, **kwargs):
        self.step_count = 0
        state, info = super().reset()
        state = self.encode_state(state)
        return Observation(state, None, False, False, info)

    def get_goal_threshold(self):
        return 7.9

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
            row, col, pass_loc, dest_idx = self.decode(state)
            direction = random.choice(['up', 'down'])  # Only noise on the y-axis due to blockages

            if direction == 'up' and row > 0:
                row -= 1
            elif direction == 'down' and row <= 4 - 1:  # Index 4 is maximum
                row += 1

            state = self.encode(row, col, pass_loc, dest_idx)

        return state


def binary_array_to_int(binary_array):
    decimal_number = 0
    for bit in binary_array:
        decimal_number = (decimal_number << 1) | bit
    return decimal_number


def get_perfect_action(taxi_row, taxi_col, passenger_location, destination):
    position = taxi_row * 5 + taxi_col

    if destination == 0:
        destination_position = 0
    elif destination == 1:
        destination_position = 4
    elif destination == 2:
        destination_position = 20
    else:
        destination_position = 23

    # 0: Move south (down)
    # 1: Move north (up)
    # 2: Move east (right)
    # 3: Move west (left)
    best_ways = [  # Red
        {0: 0, 1: 3, 2: 0, 3: 0, 4: 0,
         5: 1, 6: 3, 7: 0, 8: 0, 9: 0,
         10: 1, 11: 3, 12: 3, 13: 3, 14: 3,
         15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
         20: 1, 21: 1, 22: 1, 23: 1, 24: 1},
        # Green
        {0: 0, 1: 0, 2: 2, 3: 2, 4: 2,
         5: 0, 6: 0, 7: 2, 8: 2, 9: 1,
         10: 2, 11: 2, 12: 2, 13: 2, 14: 1,
         15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
         20: 1, 21: 1, 22: 1, 23: 1, 24: 1},
        # Yellow
        {0: 0, 1: 3, 2: 0, 3: 3, 4: 3,
         5: 0, 6: 3, 7: 0, 8: 3, 9: 3,
         10: 0, 11: 3, 12: 3, 13: 3, 14: 3,
         15: 0, 16: 1, 17: 3, 18: 1, 19: 3,
         20: 0, 21: 1, 22: 3, 23: 1, 24: 3},
        # Blue
        {0: 2, 1: 0, 2: 2, 3: 0, 4: 3,
         5: 2, 6: 0, 7: 2, 8: 0, 9: 3,
         10: 2, 11: 2, 12: 2, 13: 0, 14: 3,
         15: 1, 16: 1, 17: 1, 18: 0, 19: 3,
         20: 1, 21: 1, 22: 1, 23: 0, 24: 3}]

    # Passenger in taxi
    if passenger_location == 4:
        if position == destination_position:
            return 5

        return best_ways[destination][position]
    else:
        if passenger_location == 0:
            passenger_position = 0
        elif passenger_location == 1:
            passenger_position = 4
        elif passenger_location == 2:
            passenger_position = 20
        else:
            passenger_position = 23

        if position == passenger_position:
            return 4

        return best_ways[passenger_location][position]


if __name__ == "__main__":
    env = TaxiEnv(encoding_type='binary', det_prob_action=1.00, det_prob_state=1.00)
    observation = env.reset()

    reward = 0
    rewards = []
    eval_runs = 30
    successful_episodes = 0
    min_successful_eval_episodes = 100000
    max_successful_eval_episodes = -100000

    for _ in range(100):
        successful_episodes_eval_run = 0
        for _ in range(eval_runs):
            while True:
                observation = env.step(get_perfect_action(*list(env.decode(binary_array_to_int(observation.state)))))
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
