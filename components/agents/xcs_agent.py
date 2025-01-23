import numpy as np

from xcsf import XCS
from components.agents.abstract_agent import AbstractAgent


class XcsAgent(AbstractAgent):
    def __init__(self, gamma, epsilon, env, e0=0.01, beta=0.1, pop_size=2000):
        super(XcsAgent, self).__init__(env)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.e0 = e0
        self.beta = beta
        self.pop_size = pop_size

        self.old_state = None
        self.old_action = None

        condition = {
            "type": "ternary",
            "args": {
                "bits": 1,
                "p_dontcare": 0.5,
            }
        }

        self.model = XCS(
            x_dim=env.get_state_dim(),
            y_dim=1,
            n_actions=env.get_action_dim(),
            gamma=gamma,
            e0=e0,
            beta=beta,
            pop_size=pop_size,
            condition=condition,
            pop_init=True,
            random_state=21,
            omp_num_threads=1,
            prediction={
                "type": "constant",
            }
        )

    def get_agent_info(self):
        return {'Agent name': "XCS Agent",
                'Epsilon': self.epsilon,
                'Gamma': self.gamma,
                'e0': self.e0,
                'Beta': self.beta,
                'Population size': self.pop_size}

    def get_action(self, obs):
        self.old_state = obs.state
        if np.random.uniform(0, 1) <= self.epsilon:
            self.old_action = self.model.decision(np.array(self.old_state), True)
        else:
            self.old_action = self.model.decision(np.array(self.old_state), False)

        return self.old_action

    def update(self, obs):
        self.model.update(obs.reward, obs.terminated)

    def save_model(self, path):
        self.model.save(path + ".bin")

    def load_model(self, path):
        self.model.load(path + ".bin")
