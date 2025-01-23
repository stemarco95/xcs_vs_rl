from components.utils.utility_functions import handle_step_for_xcs, handle_trail_for_xcs


class TrainRunner:
    def __init__(self, env, agent):
        self.agent = agent
        self.env = env

    def run(self):
        scores = []
        for episode in range(self.env.train_episodes):
            if (episode + 1) % 20 == 0:
                self.agent.update_target_model()

            scores.append(self.run_episode())

        return scores

    def run_episode(self):
        score = 0
        obs = self.env.reset()
        handle_trail_for_xcs(self.agent, start=True)

        while True:
            handle_step_for_xcs(self.agent, start=True)
            action = self.agent.get_action(obs)
            obs = self.env.step(action)
            score += obs.reward
            self.agent.update(obs)
            handle_step_for_xcs(self.agent, start=False)

            if obs.terminated or obs.truncated:
                break

        handle_trail_for_xcs(self.agent, start=False)
        return score
