import json
import torch
import random
import numpy as np

from components.agents.xcs_agent import XcsAgent


def handle_trail_for_xcs(agent, start=True):
    if isinstance(agent, XcsAgent):
        if start:
            agent.model.init_trial()
        else:
            agent.model.end_trial()
    pass


def handle_step_for_xcs(agent, start=True):
    if isinstance(agent, XcsAgent):
        if start:
            agent.model.init_step()
        else:
            agent.model.end_step()
    pass


def handle_command_line(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise json.JSONDecodeError


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_one_hot_state(state, state_dim):
    one_hot_array = np.zeros(state_dim)
    one_hot_array[state] = 1
    return one_hot_array


def get_binary_state(state, state_dim):
    binary_string = np.binary_repr(state, state_dim)
    binary_array = np.array([int(bit) for bit in binary_string])
    return binary_array


def get_gray_code_state(state, state_dim):
    binary_state = get_binary_state(state, state_dim)
    return binary_state ^ (binary_state >> 1)
