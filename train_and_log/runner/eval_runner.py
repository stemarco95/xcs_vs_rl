from components.utils.utility_functions import handle_step_for_xcs, handle_trail_for_xcs


class EvalRunner:
    def __init__(self, env, agent, episodes):
        self.env = env
        self.agent = agent
        self.episodes = episodes

    def run(self):
        epsilon = self.agent.epsilon
        self.agent.epsilon = 0

        score_list = []
        steps_per_episode = []
        succeeded_episodes = []

        for episode in range(self.episodes):
            successful_episode, ep_score, ep_steps = self.run_episode()
            score_list.append(ep_score)
            steps_per_episode.append(ep_steps)
            succeeded_episodes.append(successful_episode)

        self.agent.epsilon = epsilon
        return score_list, steps_per_episode, succeeded_episodes

    def run_episode(self):
        episode_score: float = 0
        episode_steps: int = 0

        obs = self.env.reset()
        handle_trail_for_xcs(self.agent, start=True)
        while True:
            handle_step_for_xcs(self.agent, start=True)
            action = self.agent.get_action(obs)
            obs = self.env.step(action)
            handle_step_for_xcs(self.agent, start=False)
            episode_steps += 1
            episode_score += obs.reward

            if obs.terminated or obs.truncated:
                success = obs.info['success']
                break

        handle_trail_for_xcs(self.agent, start=False)
        return success, episode_score, episode_steps
