#!/usr/bin/env python3

from scipy.sparse import dok_matrix
import numpy as np
import torch
from log_controller import *

logger = Logger()
logger.set_verbosity(0)

class Encoder():
    def __init__(self):
        self.state_encoding = {}
        self.action_encoding = {}
        self.state_decoding = {}
        self.action_decoding = {}
        self.next_state_id = 0
        self.next_action_id = 0

    def encode_state(self, state):
        state = str(state)
        if state not in self.state_encoding:
            self.state_encoding[state] = self.next_state_id
            self.state_decoding[self.next_state_id] = state
            self.next_state_id += 1
        return self.state_encoding[state]

    def encode_action(self, action):

        if action not in self.action_encoding:
            if isinstance(action, (int, float)):
                if int(action) == action:
                    self.action_encoding[action] = int(action)
                    self.action_decoding[int(action)] = action

            else:
                self.action_encoding[action] = self.next_action_id
                self.action_decoding[self.next_action_id] = action
                self.next_action_id += 1

        return self.action_encoding[action]

    def encode_states(self, states):
        return [self.encode_state(state) for state in states]

    def encode_transtitions(self, transitions, sparse = False):
        n_states = self.next_state_id
        n_actions = self.next_action_id
        if sparse:
            encoded_transitions = dok_matrix((n_states*n_actions, n_states), dtype = np.float32)
        else:
            encoded_transitions = np.zeros((n_states*n_actions, n_states), dtype = np.float32)

        for (state, action, next_state), probability in transitions.items:
            state_idx = state_encoding[state]
            action_idx = action_encoding[action]
            next_state_idx = state_encoding[next_state]
            encoded_transitions[state_idx, action_idx, next_state_idx] = probability

        if sparse:
            return encoded_transitions.tocsc()
        return encoded_transitions

    def decode_state(self, encoded_state):
        encoded_state = encoded_state
        return self.state_decoding[encoded_state]

    def decode_states(self, encoded_states):
        return [self.decode_state(state) for state in encoded_states]

    def decode_action(self, encoded_action):
        return self.action_decoding[encoded_action]

    def decode_states(self, encoded_actions):
        return [self.decode_action(action) for action in encoded_actions]

def dok_to_sparse_tensor(dok_mat):
    coo = dok_mat.tocoo()
    indices = torch.tensor(np.array([coo.row, coo.col]), dtype = torch.long)
    values = torch.tensor(coo.data, dtype = torch.float32)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)
    return sparse_tensor

def is_unbounded(interval, threshold=1e30):
    return np.any(np.abs(interval) >= threshold)

def get_discretized_state(state, discretized_state_vars):
    discretized_state = []
    n_state_vars = state.shape[0]
    # Determine what state we are in
    for state_var_idx in range(n_state_vars):
        bounds = np.array(discretized_state_vars[state_var_idx], dtype = np.float32)
        le_ub = round(state[state_var_idx], 3) <= bounds[:, 1] # lt or eq to lb
        ge_lb = round(state[state_var_idx], 3) >= bounds[:, 0] # gt or eq to ub
        state_var_bin = bounds[np.where(le_ub & ge_lb)][0]
        discretized_state.append(list(state_var_bin))

    return tuple(discretized_state)

def select_action(env_model, model_state, policy, epsilon = 0):
    
    # Get encoding for model state
    encoded_state = env_model.encoder.encode_state(model_state)

    # Get list of encoded available actions for model state
    encoded_actions = [env_model.encoder.encode_action(a) for a in env_model.actions(model_state)]
    
    # Epsilon-Greedy selection
    if np.random.rand() < epsilon:
        # Randomly select action
        encoded_action = np.random.choice(encoded_actions)

    else:
        # Select action according to policy
        encoded_action = np.random.choice(encoded_actions, p = policy[encoded_state, :].numpy())

    # Decode and return action
    action = env_model.encoder.decode_action(encoded_action)

    return action

