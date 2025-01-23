import torch
import random
import numpy as np
import collections

from components.utils.networks import DQNetwork
from components.agents.abstract_agent import AbstractAgent

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


class DqnAgent(AbstractAgent):
    def __init__(self, alpha, gamma, epsilon, env):
        super(DqnAgent, self).__init__(env)

        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = 64

        self.q_network = DQNetwork(env.get_state_dim(), env.get_action_dim())
        self.target_network = DQNetwork(env.get_state_dim(), env.get_action_dim())
        self.update_target_model()

        self.alpha = alpha
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha)
        self.replay_memory = collections.deque(maxlen=20_000)

        self.old_state = None
        self.old_action = None

    def get_agent_info(self):
        return {'Agent name': "Deep Q Network Agent",
                'Alpha': self.alpha,
                'Gamma': self.gamma,
                'Epsilon': self.epsilon,
                'Batch Size': self.batch_size}

    def get_action(self, obs):
        self.old_state = obs.state.copy()
        if random.random() <= self.epsilon:
            self.old_action = random.choice(self.env.get_actions())
        else:
            with torch.no_grad():
                all_row_values = self.q_network(torch.tensor(obs.state).to(torch.float).unsqueeze(0)).tolist()
                self.old_action = np.argmax(all_row_values[0])

        return self.old_action

    def update(self, obs):
        self.replay_memory.append([self.old_state, self.old_action, obs.reward, obs.state, obs.terminated])

        if len(self.replay_memory) >= self.batch_size:
            minibatch = random.sample(self.replay_memory, self.batch_size)
            old_states, old_actions, rewards, current_states, terminals = zip(*minibatch)

            # Convert everything to tensors
            old_states = torch.tensor(np.array(old_states), dtype=torch.float)
            old_actions = torch.tensor(np.array(old_actions), dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1)
            current_states = torch.tensor(np.array(current_states), dtype=torch.float)
            terminals = torch.tensor(np.array(terminals), dtype=torch.float).unsqueeze(1)

            # Calculate current Q-values (only they one, which match to the action-indices)
            current_q_values = torch.gather(self.q_network(old_states), dim=1, index=old_actions)

            # Calculate next Q-values
            next_q_values = self.target_network(current_states).max(1)[0].unsqueeze(1)

            # Calculate expected Q-values
            with torch.no_grad():
                target_q_values = rewards + (self.gamma * next_q_values * (1 - terminals))

            # Compute loss
            loss = self.criterion(current_q_values, target_q_values.detach())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        self.q_network.save_weights(str(path) + ".pth")
