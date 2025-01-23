from components.utils.utility_functions import handle_step_for_xcs, handle_trail_for_xcs


class EvalRunner:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self):
        epsilon = self.agent.epsilon
        self.agent.epsilon = 0

        score_list = []
        succeeded_episodes = 0

        for episode in range(30):  # The evaluation takes always 30 episodes
            successful_episode, ep_score, ep_steps = self.run_episode()
            succeeded_episodes += successful_episode
            score_list.append(ep_score)

        self.agent.epsilon = epsilon
        return succeeded_episodes, score_list

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
