#!/usr/bin/env python3

from utils import Encoder, Dict_to_sparse_tensors
from scipy.sparse import lil_matrix, dok_matrix 
import torch
from log_controller import *
import line_profiler
import numpy as np
import numba
from numba import njit, types
from numba.typed import Dict
import warnings
from numba.core.errors import NumbaTypeSafetyWarning, NumbaPendingDeprecationWarning
import pdb

# Suppress specific Numba warnings about unsafe casts
warnings.filterwarnings('ignore', category=NumbaTypeSafetyWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)


class EnvModel():

    # Nested dictionary numba type definitions
    inner_dict_type = types.DictType(types.uint32, types.float32[::1])
    middle_dict_type = types.DictType(types.uint16, inner_dict_type)
    inner_val_dtype = types.float32[::1]
    action_to_count_dict_type = types.DictType(types.uint16, types.uint32)

    def __init__(self, states, actions, transitions = None):

        self.states = states
        self.actions = actions
        self.max_action_dim = max([len(actions(state)) for state in self.states])
        
        self.encoder = Encoder()
        self.encoded_states = self.encoder.encode_states(states)
        
        (self.transitions, 
         self.sa_counts) = self._construct_dynamic_transitions(transitions)
        
        self.cached_probabilities = {}
        self.cached_rewards = {}

        self._logger = Logger()
        self._logger.set_verbosity(0)
    
    def _construct_dynamic_transitions(self, transitions):
        if transitions != None:
            transitions = [(str(state), 
                            str(action), 
                            str(next_state), 
                            probability,
                            reward) 
                            for state, 
                                action, 
                                next_state, 
                                probability, 
                                reward 
                            in transitions]

        return EnvModel.__construct_dynamic_transitions(
                transitions,
                Encoder._encode_state,
                Encoder._encode_action,
                self.encoder.state_encoding,
                self.encoder.state_decoding,
                self.encoder.action_encoding,
                self.encoder.action_decoding,
                self.encoder.next_id,
                EnvModel.middle_dict_type,
                EnvModel.inner_dict_type,
                EnvModel.inner_val_dtype,
                EnvModel.action_to_count_dict_type,
                EnvModel.insert_value,
                )

    


    @staticmethod
    @njit
    def __construct_dynamic_transitions(
            raw_transitions, 
            encode_state, 
            encode_action, 
            state_encoding,
            state_decoding,
            action_encoding,
            action_decoding,
            next_id,
            middle_dict_type,
            inner_dict_type,
            inner_val_dtype,
            action_to_count_dict_type,
            insert_value
            ):
        """
        Constructs a numba.typed.Dicts graph for storing and updating probabilities
        """

        # Initialize Numba-Typed-Dicts for transition dynamics and data
        transitions = Dict.empty(
                key_type = types.uint32,
                value_type = middle_dict_type
                )

        state_action_counts = Dict.empty(
                key_type = types.uint32,
                value_type = action_to_count_dict_type
                )

        # Parse transitions into numba.typed.Dict (if provided)
        if raw_transitions != None:
            count = 0
            for (state, action, next_state, probability, reward) in raw_transitions:

                # Retrieve state, action, and next_state indices
                encoded_state = encode_state(state, state_encoding, state_decoding, next_id)
                encoded_action = encode_action(action, action_encoding, action_decoding, next_id)
                encoded_next_state = encode_state(next_state, state_encoding, state_decoding, next_id)

                # Create transitinos data packet to insert as value in dictionary
                transitions_data = np.array([probability, reward, count], np.float32)

                # Assign probability and rewards values
                insert_value(transitions, inner_dict_type, inner_val_dtype, encoded_state, encoded_action, encoded_next_state, transitions_data)
        
        return transitions, state_action_counts

    @staticmethod
    @njit
    def insert_value(transitions, inner_dict_type, inner_val_dtype, state, action, next_state, value):

        if state not in transitions:

            transitions[state] = Dict.empty(
                key_type = types.uint16,
                value_type = inner_dict_type
                )

        if action not in transitions[state]:
            transitions[state][action] = Dict.empty(
                key_type = types.uint32,
                value_type = inner_val_dtype
                )

        transitions[state][action][next_state] = value
    
    @staticmethod
    @njit
    def insert_state_action_count(state_action_counts, state, action, value):

        if state not in state_action_counts:
            state_action_counts[state] = Dict.empty(
                    key_type=types.uint16,
                    value_type=types.uint32
                    )
        state_action_counts[state][action] = value

    def update_transitions(self, params, verbosity = 0):
        """
        This function invalidates the cached probabilities and rewards and 
        serves as a public interface for the private njitted static method 
        __update_transitions.
        """

        # Invalidate cached probabilities and rewards
        self.cached_probabilities = {}
        self.cached_rewards = {}

        params = [(str(state), str(action), str(next_state), np.float32(reward))
                  for state, action, next_state, reward, in params]

        # 
        EnvModel.__update_transitions(self.transitions,
                                      self.sa_counts,
                                      Encoder._encode_state,
                                      self.encoder.state_encoding,
                                      self.encoder.state_decoding,
                                      Encoder._encode_action,
                                      self.encoder.action_encoding,
                                      self.encoder.action_decoding,
                                      self.encoder.next_id,
                                      params, 
                                      EnvModel.inner_dict_type,
                                      EnvModel.inner_val_dtype,
                                      EnvModel.insert_state_action_count,
                                      EnvModel.get_or_assign_default,
                                      EnvModel.get_sa_count,
                                      verbosity = verbosity)

    @staticmethod
    @njit
    def __update_transitions(transitions,
                             sa_counts,
                             encode_state,
                             state_encoding,
                             state_decoding,
                             encode_action,
                             action_encoding,
                             action_decoding,
                             next_id,
                             params, 
                             inner_dict_type,
                             inner_val_dtype,
                             insert_state_action_count,
                             get1,
                             get2,
                             verbosity = 0):

        """
        This function updates the dynamic transitions based on the data in 
        params = [(state, action, next_state, reward)]
        """
        
        # Loop through exploration data and update transitions model
        for state, action, next_state, reward in params:
            
            # Get transition indices
            state_idx = encode_state(state, state_encoding, state_decoding, next_id)
            action_idx = encode_action(action, action_encoding, action_decoding, next_id)
            next_state_idx = encode_state(next_state, state_encoding, state_decoding, next_id)
            
            # Get previous transition data record
            data = get1(transitions,
                        inner_dict_type,
                        inner_val_dtype,
                        state_idx,
                        action_idx,
                        next_state_idx)

            old_sa_count = get2(sa_counts,
                               state_idx,
                               action_idx)
            
            """
            The old_data has the expected structure of 
             - old_probability = old_data[0]
             - old_reward = old_data[1]
             - old_sas_count = old_data[2]
            """
        
            # Named indices for code clarity
            prob_idx = 0
            reward_idx = 1
            sas_count_idx = 2

            # Update transition count
            data[sas_count_idx] += 1

            # Update number of times action a was chosen in state s
            new_sa_count = old_sa_count + 1            
            insert_state_action_count(sa_counts, state_idx, action_idx, new_sa_count)

            # Update rewards with incremental averaging
            data[reward_idx] = data[reward_idx] + (reward - data[reward_idx])/data[sas_count_idx]

            # Update probabilities with new transition count
            for s_prime, iter_data in transitions[state_idx][action_idx].items():
                
                count = iter_data[2]
                prob = count/new_sa_count
                iter_data[0] = prob
            
            #if state != next_state and verbosity == 2:
             #   print("--------- Updating the following transition: ---------")
              #  print(f"Action_idx {action_idx}")
               # print(f"State {state_idx}")
               # print(f"Action: {action}")
               # print(f"Next_State: {next_state_idx}")
               # print(f"k_vec/ksa: {prob}")
               # print(f"Reward: {exp_reward}")
               # pass

#    @staticmethod
#    @njit
#    def get_nested(nested_dict, *keys):
        
#        default_value = 0

#        state, action = keys[0], keys[1]

#        if state in nested_dict and action in nested_dict[state]:
#            inner_dict = nested_dict[state][action]

#            if len(keys) == 2:
#                return inner_dict

#            elif len(keys) == 3:
#                next_state = keys[2]

#                if next_state in inner_dict:
#                    return inner_dict[next_state]

#        return default_value

    @staticmethod
    @njit
    def get_or_assign_default(nested_dict, inner_dict_type, inner_val_dtype, state, action, next_state):
        """
        Fetches a value from a nested dictionary structure using provided keys.
        Returns a default value if any key is missing and inserts it.

        Parameters:
        nested_dict (Dict): The nested dictionary to search.
        state (Hashable): The first-level key.
        action (Hashable): The second-level key.
        next_state (Hashable, optional): The optional third-level key.

        Returns:
        Any: The value from the nested dictionary or a default value if not found.
        """
        
        default_probability = 0.0
        default_exp_reward = 0.0
        default_count = 0.0

        default_value = np.array([default_probability,
                                  default_exp_reward,
                                  default_count], dtype = np.float32)

        # Check for state and action keys.
        if state not in nested_dict:
            nested_dict[state] = Dict.empty(
                    key_type = types.uint16,
                    value_type = inner_dict_type
                    )

        level2 = nested_dict[state]

        if action not in level2:
            level2[action] = Dict.empty(
                    key_type = types.uint32,
                    value_type = inner_val_dtype
                    )

        level3 = level2[action]

        if next_state not in level3:
            level3[next_state] = default_value

        return level3[next_state]

    @staticmethod
    @njit
    def get_sa_count(sa_counts, state, action):
        """
        Fetches a value from a nested dictionary structure using provided keys.
        Returns a default value if any key is missing.

        Parameters:
        sa_counts (Dict): The nested dictionary to search.
        state (Hashable): The first-level key.
        action (Hashable): The second-level key.

        Returns:
        Any: The value from the nested dictionary or a default value if not found.
        """

        default_value = 0
        
        # Check for state and action keys.
        if state in sa_counts and action in sa_counts[state]:
            return sa_counts[state][action]

        return default_value

    @staticmethod
    @njit
    def simple_get(any_dict, key):
        return any_dict.get(key)
    
    @staticmethod
    @njit
    def initialize_if_empty(transitions, inner_dict_type, encoded_state):

        if encoded_state not in transitions:
            transitions[encoded_state] = Dict.empty(
                    key_type = types.uint16,
                    value_type = EnvModel.inner_dict_type
                    )

    def get_prob_and_reward_tensors(self, encoded_state):

        return self._get_prob_and_reward_tensors(encoded_state, self.max_action_dim, len(self.states))


    def _get_prob_and_reward_tensors(self, encoded_state, n_actions, n_states):

        if encoded_state not in self.cached_probabilities:
            

            if EnvModel.simple_get(self.transitions, encoded_state) == None:
                return (torch.sparse_coo_tensor(size = (n_actions, n_states)), 
                        torch.sparse_coo_tensor(size = (n_actions, n_states)))

            prob_reward_dict = EnvModel.simple_get(self.transitions, encoded_state)

            (self.cached_probabilities[encoded_state],
             self.cached_rewards[encoded_state]) = Dict_to_sparse_tensors(prob_reward_dict, (n_actions, n_states))

        return self.cached_probabilities[encoded_state], self.cached_rewards[encoded_state]

    def child(self):
        return EnvChild(self.encoder, self.actions)

class EnvChild(EnvModel):
    
    def __init__(self, encoder, actions):

        self.encoder = encoder
        self.actions = actions

