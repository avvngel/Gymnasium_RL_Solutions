#!/usr/bin/env python3

# Import necessary modules
import numpy as np
from env_model import *
from policy import *
from utils import *
from log_controller import *

# Define States
states = list(range(10))

# Write function mapping states to available actions
def available_actions(state):
    """
    Returns the available actions for the state provided.
    """
    if state > 5:
        return [0, 1, 3, 4]
    else:
        return [2, 3, 5, 7]

# Create discretized environment model
env_model = EnvModel(states, available_actions)
encoded_actions = [env_model.encoder.encode_action(action)
                   for state in env_model.states
                   for action in available_actions(state)]
print("Encoded Actions:")
print(env_model.encoder.action_encoding)
                                                     
