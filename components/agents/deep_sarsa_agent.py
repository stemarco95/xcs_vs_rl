import torch
import random
import numpy as np
import collections

from components.agents.abstract_agent import AbstractAgent
from components.utils.networks import DQNetwork

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


class DeepSarsaAgent(AbstractAgent):
    def __init__(self, alpha, gamma, epsilon, env):
        super(DeepSarsaAgent, self).__init__(env)

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = 64

        self.q_network = DQNetwork(env.get_state_dim(), env.get_action_dim())

        self.alpha = alpha
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha)
        self.replay_memory = collections.deque(maxlen=20_000)

        self.old_action = None
        self.next_action = None
        self.old_state = None

    def get_agent_info(self):
        return {'Agent name': "Deep SARSA Agent",
                'Alpha': self.alpha,
                'Gamma': self.gamma,
                'Epsilon': self.epsilon,
                'Batch Size': self.batch_size}

    def get_action(self, obs):
        self.old_state = obs.state.copy()
        if self.next_action is None:
            self.old_action = self.get_new_action(
                torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0))
        else:
            self.old_action = self.next_action

        return self.old_action

    def get_new_action(self, state_tensor):
        if np.random.uniform(0, 1) <= self.epsilon:
            return random.choice(self.env.get_actions())
        else:
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.data.numpy())

    def update(self, obs):
        self.next_action = self.get_new_action(torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0))

        self.replay_memory.append([self.old_state, self.old_action, obs.reward, obs.state, self.next_action, obs.terminated])

        if len(self.replay_memory) >= self.batch_size:
            minibatch = random.sample(self.replay_memory, self.batch_size)
            old_states, old_actions, rewards, current_states, current_action, terminals = zip(*minibatch)

            # Convert everything to tensors
            old_states = torch.tensor(np.array(old_states), dtype=torch.float)
            old_actions = torch.tensor(np.array(old_actions), dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1)
            current_states = torch.tensor(np.array(current_states), dtype=torch.float)
            current_action = torch.tensor(np.array(current_action), dtype=torch.long).unsqueeze(1)
            terminals = torch.tensor(np.array(terminals), dtype=torch.float).unsqueeze(1)

            # Calculate current Q-values (only they one, which match to the action-indices)
            current_q_values = torch.gather(self.q_network(old_states), dim=1, index=old_actions)

            # Calculate next Q-values
            next_q_values = torch.gather(self.q_network(current_states), dim=1, index=current_action)

            # Calculate expected Q-values
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terminals))

            # Compute loss
            loss = self.criterion(current_q_values, target_q_values.detach())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if obs.terminated or obs.truncated:
            self.next_action = None

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))

    def save_model(self, path):
        self.q_network.save_weights(str(path) + ".pth")
