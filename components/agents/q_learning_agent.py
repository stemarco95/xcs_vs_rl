import pickle
import numpy as np

from collections import defaultdict
from components.agents.abstract_agent import AbstractAgent


class QLearningAgent(AbstractAgent):
    def __init__(self, alpha, gamma, epsilon, env):
        super(QLearningAgent, self).__init__(env)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.old_state = None
        self.old_action = None
        self.model = defaultdict(lambda: np.zeros(len(self.env.get_actions())))

    def get_agent_info(self):
        return {'Agent Name': "Q-Learning Agent",
                'Epsilon': self.epsilon,
                'Alpha': self.alpha,
                'Gamma': self.gamma}

    def get_action(self, obs):
        self.old_state = self.to_hashable(obs.state)

        if np.random.uniform(0, 1) <= self.epsilon:
            self.old_action = np.random.choice(self.env.get_actions())
        else:
            max_indices = np.where(self.model[self.old_state] == np.max(self.model[self.old_state]))[0]
            self.old_action = np.random.choice(max_indices)

        return self.old_action

    def update(self, obs):
        obs.state = self.to_hashable(obs.state)

        max_action = np.argmax(self.model[obs.state])
        target_value = obs.reward + self.gamma * (1 - obs.terminated) * self.model[obs.state][max_action]
        loss = target_value - self.model[self.old_state][self.old_action]
        self.model[self.old_state][self.old_action] += self.alpha * loss

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
