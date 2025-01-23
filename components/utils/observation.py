
class Observation:
    def __init__(self, obs, reward=None, terminated=None, truncated=None, info=None):
        self.state = obs
        self.info = info
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
