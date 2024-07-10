#!/usr/bin/env python3

from scipy.sparse import dok_matrix
import numpy as np
import torch
import time
from log_controller import *
from numba import njit, types
from numba.typed import List, Dict
import faulthandler
import warnings
from numba.core.errors import NumbaTypeSafetyWarning, NumbaPendingDeprecationWarning
from itertools import product
import pdb

# Suppress specific Numba warnings about unsafe casts
warnings.filterwarnings('ignore', category=NumbaTypeSafetyWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)


logger = Logger()
logger.set_verbosity(0)

class Encoder():

    def __init__(self):
        self.state_encoding = Encoder.initialize_state_encoding()
        self.action_encoding = Encoder.initialize_action_encoding()
        self.state_decoding = Encoder.initialize_state_decoding()
        self.action_decoding = Encoder.initialize_action_decoding()
        self.next_id = Encoder.initialize_next_id()
    
    def encode_state(self, state):

        state = str(state)

        return Encoder._encode_state(
               state, 
               self.state_encoding, 
               self.state_decoding, 
               self.next_id
               )

    def encode_action(self, action):
        
        action = str(action)
        return Encoder._encode_action(
                action, 
                self.action_encoding, 
                self.action_decoding, 
                self.next_id
                )

    def encode_states(self, states):
        return [self.encode_state(state) for state in states]

    def decode_state(self, encoded_state):
        return Encoder.get(self.state_decoding, encoded_state)

    def decode_states(self, encoded_states):
        return [self.decode_state(state) for state in encoded_states]

    def decode_action(self, encoded_action):
        return Encoder.get(self.action_decoding, encoded_action)

    def decode_actions(self, encoded_actions):
        return [self.decode_action(action) for action in encoded_actions]

    @staticmethod
    @njit
    def get(numba_Dict, key):
        return numba_Dict[key]
    
    @staticmethod
    @njit
    def initialize_state_encoding():
        return Dict.empty(types.unicode_type, types.uint32)

    @staticmethod
    @njit
    def initialize_action_encoding():
        return Dict.empty(types.unicode_type, types.uint16)

    @staticmethod
    @njit
    def initialize_state_decoding():
        return Dict.empty(types.uint32, types.unicode_type)

    @staticmethod
    @njit
    def initialize_action_decoding():
        return Dict.empty(types.uint16, types.unicode_type)

    @staticmethod
    @njit
    def initialize_next_id():

        next_id = Dict.empty(types.unicode_type, types.uint32)

        next_id['state'] = 0
        next_id['action'] = 0

        return next_id

    @staticmethod
    @njit
    def _encode_state(state, state_encoding, state_decoding, next_id):

        if state not in state_encoding:

            state_encoding[state] = next_id['state']
            state_decoding[next_id['state']] = state
            next_id['state'] += 1

        return state_encoding[state]
    
    @staticmethod
    @njit
    def _encode_action(action, action_encoding, action_decoding, next_id):

        if action not in action_encoding:

            action_encoding[action] = next_id['action']
            action_decoding[next_id['action']] = action
            next_id['action'] += 1

        return action_encoding[action]


def dok_to_sparse_tensor(dok_mat):

    coo = dok_mat.tocoo()
    indices = torch.tensor(np.array([coo.row, coo.col]), dtype = torch.long)
    values = torch.tensor(coo.data, dtype = torch.float32)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)

    return sparse_tensor

def Dict_to_sparse_tensors(typed_Dict, shape):
    indices, probability_values, exp_reward_values = get_coo_from_transitions(typed_Dict)
    indices = torch.tensor(indices, dtype = torch.long)
    probability_values = torch.tensor(probability_values, dtype = torch.float32)
    exp_reward_values = torch.tensor(exp_reward_values, dtype = torch.float32)
    probability_sparse_tensor = torch.sparse_coo_tensor(indices, probability_values, torch.Size(shape))
    exp_reward_sparse_tensor = torch.sparse_coo_tensor(indices, exp_reward_values, torch.Size(shape))

    return probability_sparse_tensor, exp_reward_sparse_tensor

@njit
def get_coo_from_transitions(transitions):

    row_indices = List.empty_list(types.uint64)
    col_indices = List.empty_list(types.uint64)
    probability_values = List.empty_list(types.float32)
    exp_reward_values = List.empty_list(types.float32)

    for row_idx, col_and_val in transitions.items():
        if len(col_and_val) == 0:
            break
        for col_idx, value in col_and_val.items():
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            probability_values.append(value[0])
            exp_reward_values.append(value[1])
    
    indices = [row_indices,
               col_indices]

    return indices, probability_values, exp_reward_values




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
        p = policy[encoded_state, :].numpy()
        encoded_action = np.random.choice(encoded_actions, p = p)

    # Decode and return action
    action = env_model.encoder.decode_action(encoded_action)

    return int(action)


def explore(env, env_child, pi, epsilon, discretized_state_vars, n_episodes, max_time_steps, result_queue, seed=None):

    # Empty list to store transition update info
    params = []

    # Pre-allocate array for storing rewards and computing episodic average
    episode_rewards = np.zeros(n_episodes)

    # Exploration loop
    for episode in range(n_episodes):

        logger('')
        logger('')
        logger(f"#************************* EPISODE {episode + 1} ******************************#")

        # Reset environment to initial state
        if seed == None:
            env_state = env.reset()[0]
        else:
            env_state = env.reset(seed = seed)[0]

        for t in range(max_time_steps):

            # Convert continuous environment state into discrete model state
            model_state = get_discretized_state(env_state, discretized_state_vars)

            # Select action based on policy
            action = select_action(env_child, model_state, pi, epsilon = epsilon)

            # Execute action and record environment reaction
            env_state, reward, terminated, truncated, info = env.step(action)

            # Convert continuous next environment state in to discrete next model state
            next_model_state = get_discretized_state(env_state, discretized_state_vars)

            # Add reward_shaping
            distance_from_center = abs(env_state[0])
            reward = reward - distance_from_center/2.4

            # Debugging output
            logger(f"Discretized State: {env_child.encoder.encode_state(model_state)}")
            logger(f"Action taken: {action}")
            logger(f"Discretized Next State: {env_child.encoder.encode_state(next_model_state)}")

            # Check for episode termination or truncation
            if truncated or terminated:

                reward = -1000
                params.append((model_state, action, next_model_state, reward))
                break

            # record_rewards
            episode_rewards[episode] += 1

            # Update transition probabilities and expected rewards
            params.append((model_state, action, next_model_state, reward))

    result_queue.put(params)
    print(f"Average reward: {episode_rewards.mean()}")

def discretize_state_variables(global_bounds, episode_bounds, N):
    
    # Discretize state variables
    n_state_vars = global_bounds.shape[1]
    discretized_state_vars = []

    for state_var_idx in range(n_state_vars):

        # Get episode termination bounds
        lb = episode_bounds[:, state_var_idx][0]
        ub = episode_bounds[:, state_var_idx][1]


        # Set bound based for infinite termination windows
        if is_unbounded([lb, ub]):
            lb = -5
            ub = 5

        # Set global bounds
        left_bin = [[global_bounds[0, state_var_idx], round(lb, 2)]]
        right_bin = [[round(ub, 2), global_bounds[1, state_var_idx]]]

        # Termination window discretization
        bin_width = (ub - lb)/N
        middle_bins = [[round(lb+bin_width*i, 2),
                        round(lb+(bin_width)*(i+1), 2)]
                        for i in range(N)]

        # Store discretized state variables
        discretized_state_vars += [left_bin + middle_bins + right_bin]

    return discretized_state_vars



def display_parsed_args(args, display_width, logger, verbosity):
    left_width = int(display_width/2)
    right_width = int(display_width/2)
    logger("", msg_verbosity = verbosity)
    logger("-"*display_width, msg_verbosity = verbosity)
    logger("Parsing command line arguments and assigning defaults...".center(display_width), msg_verbosity = verbosity)
    logger("-"*display_width, msg_verbosity = verbosity)
    logger("N".ljust(left_width, ".") + str(args.N).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("n_epochs".ljust(left_width, ".") + str(args.n_epochs).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("n_training_episodes".ljust(left_width, ".") + format(args.n_training_episodes, ",").rjust(right_width, "."), msg_verbosity = verbosity)
    logger("epsilon".ljust(left_width, ".") + str(args.epsilon).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("gamma".ljust(left_width, ".") + str(args.gamma).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("epsilon_decay_rate".ljust(left_width, ".") + str(args.epsilon_decay_rate).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("seed".ljust(left_width, ".") + str(args.seed).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("n_cores".ljust(left_width, ".") + str(args.n_cores).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("verbosity".ljust(left_width, ".") + str(args.v).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("demonstrate".ljust(left_width, ".") + str(args.demonstrate).rjust(right_width, "."), msg_verbosity = verbosity)
    logger("-"*display_width, msg_verbosity = verbosity)

