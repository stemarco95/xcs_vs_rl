import pickle
import random
import numpy as np

from collections import defaultdict
from components.agents.abstract_agent import AbstractAgent


class SarsaAgent(AbstractAgent):
    def __init__(self, alpha, gamma, epsilon, env):
        super(SarsaAgent, self).__init__(env)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.old_state = None
        self.old_action = None
        self.next_action = None
        self.model = defaultdict(lambda: np.zeros(len(self.env.get_actions())))

    def get_agent_info(self):
        return {'Agent Name': "SARSA Agent",
                'Epsilon': self.epsilon,
                'Alpha': self.alpha,
                'Gamma': self.gamma}

    def get_action(self, obs):
        obs.state = self.to_hashable(obs.state)
        self.old_state = obs.state
        if self.next_action is None:
            self.old_action = self.get_new_action(obs)
        else:
            self.old_action = self.next_action

        return self.old_action

    def get_new_action(self, obs):
        if random.uniform(0, 1) <= self.epsilon:
            return np.random.choice(self.env.get_actions())
        else:
            # If more than one max value exist, we take a random index of them
            max_indices = np.where(self.model[obs.state] == np.max(self.model[obs.state]))[0]
            return np.random.choice(max_indices)

    def update(self, obs):
        obs.state = self.to_hashable(obs.state)

        self.next_action = self.get_new_action(obs)
        current_q_value = self.model[self.old_state][self.old_action]
        next_q_value = self.model[obs.state][self.next_action]
        target_q_value = obs.reward + self.gamma * next_q_value * (1 - obs.terminated)
        loss = target_q_value - current_q_value
        self.model[self.old_state][self.old_action] += self.alpha * loss

        # If the agent reached the goal, he has to ignore the selected action
        if obs.terminated or obs.truncated:
            self.next_action = None

    @staticmethod
    def to_hashable(obj):
        if isinstance(obj, (tuple, frozenset)):
            return obj
        elif isinstance(obj, dict):
            return tuple(sorted(obj.items()))
        elif isinstance(obj, (list, set)):
            return tuple(obj)
        elif isinstance(obj, np.ndarray):
            return tuple([i for i in obj])
        else:
            return obj

    def load_model(self, path):
        with open(path + '.pkl', 'rb') as file:
            model_dict = pickle.load(file)
            self.model = defaultdict(lambda: np.zeros(len(self.env.get_actions())))
            for key, value in model_dict.items():
                self.model[key] = np.array(value)

    def save_model(self, path):
        with open(path + '.pkl', 'wb') as file:
            pickle.dump(dict(self.model), file)
