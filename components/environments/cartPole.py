import math
import random
import numpy as np

from components.utils.observation import Observation
from components.environments.abstract_env import AbstractEnv
from gymnasium.envs.classic_control.cartpole import CartPoleEnv as GymCartPoleEnv
from components.utils.utility_functions import get_one_hot_state, get_binary_state, get_gray_code_state


# https://gymnasium.farama.org/environments/classic_control/cart_pole/
class CartPoleEnv(GymCartPoleEnv, AbstractEnv):
    def __init__(self, render_mode=None, iterations=25, train_episodes=30, det_prob_state=1.0, det_prob_action=1.0,
                 encoding_type='decimal', discretization_bins=10, seed=21):
        GymCartPoleEnv.__init__(self, render_mode=render_mode)
        AbstractEnv.__init__(self, iterations=iterations, train_episodes=train_episodes, max_payoff=1.0)

        self.det_prob_state = det_prob_state
        self.det_prob_action = det_prob_action
        self.encoding_type = encoding_type
        self.set_seed(seed)
        self.step_count = 0
        self.discretization_bins = discretization_bins

        self.bins = [np.linspace(-2.4, 2.4, discretization_bins),
                     np.linspace(-4, 4, discretization_bins),
                     np.linspace(-.2095, .2095, discretization_bins),
                     np.linspace(-4, 4, discretization_bins)]

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        super().reset(seed=seed)

    def get_env_info(self):
        return {"Env name": "Cart Pole Environment",
                "Number of iterations": self.iterations,
                "Number of training episodes per iteration": self.train_episodes,
                "Deterministic action probability": self.det_prob_action,
                "Deterministic state probability": self.det_prob_state}

    def get_state_dim(self):
        if self.encoding_type == 'one_hot':
            return self.discretization_bins ** 4
        if self.encoding_type == 'gray_code' or self.encoding_type == 'binary':
            return math.ceil(math.log2(self.discretization_bins ** 4))

        return 4  # decimal

    def get_action_dim(self):
        return 2

    def get_actions(self):
        # 0: Push cart to the left, 1: Push cart to the right
        return [0, 1]

    def step(self, action):
        action = self.choose_action_with_probability(action)
        obs = Observation(*super().step(action))
        obs.state = self.make_noisy_state_if_necessary(obs.state)
        if self.discretization_bins > 0:
            obs.state = self.discretize(obs.state)
        obs.state = self.encode_state(obs.state)
        self.step_count += 1
        if self.step_count == 500:
            obs.truncated = True

        obs.info['success'] = True if obs.truncated else False
        return obs

    def reset(self, **kwargs):
        state, info = super().reset()
        if self.discretization_bins > 0:
            state = self.discretize(state)
        self.step_count = 0

        return Observation(self.encode_state(state), None, False, False, info)

    def get_goal_threshold(self):
        return 500

    def choose_action_with_probability(self, current_action):
        if random.random() <= self.det_prob_action:
            return current_action
        else:
            return random.choice([0, 1])

    def discretize(self, state):
        state[0] = np.digitize(state[0], self.bins[0])
        state[1] = np.digitize(state[1], self.bins[1])
        state[2] = np.digitize(state[2], self.bins[2])
        state[3] = np.digitize(state[3], self.bins[3])

        return state

    def encode_state(self, state):
        single_state = 0
        for i, value in enumerate(state.astype(int)):
            single_state += value * (10 ** (3 - i))

        if self.encoding_type == 'one_hot':
            return get_one_hot_state(single_state, self.get_state_dim())
        if self.encoding_type == 'binary':
            return get_binary_state(single_state, self.get_state_dim())
        if self.encoding_type == 'gray_code':
            return get_gray_code_state(single_state, self.get_state_dim())

        return state  # decimal

    def make_noisy_state_if_necessary(self, state):
        if random.random() > self.det_prob_state:
            noise_scale = 1
            dimensions_to_noise = np.random.choice(4, 2, replace=False)
            for dimension in dimensions_to_noise:
                noise_direction = np.random.choice([-1, 1])
                noise = (self.bins[dimension][1] - self.bins[dimension][0]) * noise_scale * noise_direction
                state[dimension] += noise
                state[dimension] = np.clip(state[dimension], self.bins[dimension][0], self.bins[dimension][-1])
        return state
