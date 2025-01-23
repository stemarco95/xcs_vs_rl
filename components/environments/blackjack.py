import random
import numpy as np

from components.utils.observation import Observation
from components.environments.abstract_env import AbstractEnv
from gymnasium.envs.toy_text.blackjack import BlackjackEnv as GymBlackjackEnv
from components.utils.utility_functions import get_gray_code_state


class BlackjackEnv(GymBlackjackEnv, AbstractEnv):
    def __init__(self, render_mode=None, natural=False, encoding_type='decimal', iterations=25, train_episodes=30, det_prob_state=1.0, seed=22):
        GymBlackjackEnv.__init__(self, render_mode=render_mode, natural=natural)
        AbstractEnv.__init__(self, iterations=iterations, train_episodes=train_episodes, max_payoff=1.5 if natural else 1.0)

        self.encoding_type = encoding_type
        self.natural = natural
        self.det_prob_state = det_prob_state
        self.set_seed(seed)

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        super().reset(seed=seed)

    def get_env_info(self):
        return {"Env name": "Blackjack Environment",
                "Number of iterations": self.iterations,
                "Number of training episodes per iteration": self.train_episodes,
                "Deterministic state probability": self.det_prob_state}

    def get_state_dim(self):
        if self.encoding_type == 'decimal':
            return 3
        if self.encoding_type == 'gray_code':
            return 9

    def get_action_dim(self):
        return 2

    def get_actions(self):
        return [0, 1]

    def step(self, action):
        obs = Observation(*super().step(action))
        obs.state = np.array(obs.state)
        obs.state = self.make_noisy_state_if_necessary(obs.state)
        obs.state = self.encode_state(obs.state)
        obs.info['success'] = True if obs.reward > 0 else False
        return obs

    def reset(self, **kwargs):
        state, info = super().reset()
        state = self.encode_state(state)
        state = np.array(state)
        return Observation(state, None, False, False, info)

    def encode_state(self, state):
        if self.encoding_type == 'gray_code':
            state = (state[0] - 4) * 20 + (state[1] - 2) * 2 + state[2]  # encode to one decimal
            return get_gray_code_state(state, self.get_state_dim())

        return state  # decimal

    def get_goal_threshold(self):
        if self.natural:
            return -0.035
        else:
            return -0.055

    def make_noisy_state_if_necessary(self, state):
        if random.random() > self.det_prob_state:
            # direction = random.choice(['up', 'down'])
            value = random.choice([1, 2])

            if state[0] - value >= 4:  # The minimal current player sum ist 4
                state[0] = state[0] - value
            elif state[0] - (value - 1) >= 4:
                state[0] = state[0] - (value - 1)

        return state


def basic_strategy(player_sum, dealer_card, usable_ace):
    # Hard hands
    if not usable_ace:
        if player_sum >= 17:
            return 0  # Stand
        elif 13 <= player_sum <= 16:
            if dealer_card <= 6:
                return 0  # Stand
            else:
                return 1  # Hit
        elif player_sum == 12:
            if 4 <= dealer_card <= 6:
                return 0  # Stand
            else:
                return 1  # Hit
        elif player_sum == 11:
            return 1  # Double down (treated as hit in this simulation)
        elif player_sum == 10:
            if dealer_card <= 9:
                return 1  # Double down (treated as hit in this simulation)
            else:
                return 1  # Hit
        elif player_sum == 9:
            if 3 <= dealer_card <= 6:
                return 1  # Double down (treated as hit in this simulation)
            else:
                return 1  # Hit
        else:
            return 1  # Hit
    # Soft hands
    else:
        if player_sum >= 19:
            return 0  # Stand
        elif player_sum == 18:
            if dealer_card >= 9 or dealer_card == 1:
                return 1  # Hit
            else:
                return 0  # Stand
        elif player_sum == 17 or player_sum == 16:
            return 1  # Double down (treated as hit in this simulation)
        elif player_sum == 15 or player_sum == 14:
            if 5 <= dealer_card <= 6:
                return 1  # Double down (treated as hit in this simulation)
            else:
                return 1  # Hit
        elif player_sum == 13:
            if 4 <= dealer_card <= 6:
                return 1  # Double down (treated as hit in this simulation)
            else:
                return 1  # Hit
        elif player_sum == 12:
            if 4 <= dealer_card <= 6:
                return 1  # Double down (treated as hit in this simulation)
            else:
                return 1  # Hit
        else:
            return 1  # Hit


if __name__ == "__main__":
    env = BlackjackEnv(det_prob_state=1.0, encoding_type='decimal', natural=True)
    observation = env.reset()

    reward = 0
    rewards = []
    eval_runs = 30
    successful_episodes = 0
    min_successful_eval_episodes = 100000
    max_successful_eval_episodes = -100000

    for _ in range(10000):
        successful_episodes_eval_run = 0
        for _ in range(eval_runs):
            while True:
                observation = env.step(basic_strategy(*observation.state))
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
