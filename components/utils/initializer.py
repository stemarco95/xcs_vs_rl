from components.agents.dqn_agent import DqnAgent
from components.agents.xcs_agent import XcsAgent
from components.agents.sarsa_agent import SarsaAgent
from components.agents.q_learning_agent import QLearningAgent
from components.agents.deep_sarsa_agent import DeepSarsaAgent

from components.environments.taxi import TaxiEnv
from components.environments.frozenLake import FrozenLakeEnv
from components.environments.blackjack import BlackjackEnv
from components.environments.cartPole import CartPoleEnv
from components.environments.cliffWalking import CliffWalkingEnv


def get_environment_instance(env_config):
    env = None

    name = env_config.get("type").lower()
    if name == "taxi":
        env = TaxiEnv(**env_config['parameter'])
    elif name == "blackjack":
        env = BlackjackEnv(**env_config['parameter'])
    elif name == "cartpole":
        env = CartPoleEnv(**env_config['parameter'])
    elif name == "frozenlake":
        env = FrozenLakeEnv(**env_config['parameter'])
    elif name == "cliffwalking":
        env = CliffWalkingEnv(**env_config['parameter'])

    return env


def get_agent_instance(agent_config, env):
    agent = None

    agent_config['parameter']['env'] = env
    agent_type = agent_config.get("type")
    if agent_type == "dqn":
        agent = DqnAgent(**agent_config['parameter'])
    elif agent_type == "q_learning":
        agent = QLearningAgent(**agent_config['parameter'])
    elif agent_type == "deep_sarsa":
        agent = DeepSarsaAgent(**agent_config['parameter'])
    elif agent_type == "sarsa":
        agent = SarsaAgent(**agent_config['parameter'])
    elif agent_type == "xcs":
        agent = XcsAgent(**agent_config['parameter'])

    return agent


def get_instances(config_env, config_agent):
    env = get_environment_instance(config_env)
    agent = get_agent_instance(config_agent, env)

    return env, agent
