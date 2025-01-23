from abc import abstractmethod


class AbstractAgent:
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_action(self, obs):
        ...

    @abstractmethod
    def update(self, obs):
        ...

    @abstractmethod
    def get_agent_info(self):
        ...

    @abstractmethod
    def save_model(self, path):
        ...

    def update_target_model(self):
        pass
