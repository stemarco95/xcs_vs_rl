from abc import abstractmethod


class AbstractEnv:
    def __init__(self, iterations, train_episodes, max_payoff):
        self.iterations = iterations
        self.train_episodes = train_episodes
        self.max_payoff = max_payoff

    @abstractmethod
    def set_seed(self, seed):
        pass

    @abstractmethod
    def get_env_info(self):
        pass

    @abstractmethod
    def get_state_dim(self):
        pass

    @abstractmethod
    def get_action_dim(self):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_goal_threshold(self):
        ...
