#!/usr/bin/env python3

from utils import Encoder, dok_to_sparse_tensor
from scipy.sparse import lil_matrix, dok_matrix 
import torch
from log_controller import *


class EnvModel():

    def __init__(self, states, actions, transitions = None):
        self.states = states
        self.actions = actions
        self.max_action_dim = max([len(actions(state)) for state in self.states])
        self.raw_transitions = transitions
        self.encoder = Encoder()
        self.encoded_states = self.encoder.encode_states(states)
        self.dynamic_transitions = self._construct_dynamic_transitions()
        self.cached_probabilities = {}
        self.cached_rewards = {}
        self._logger = Logger()
        self._logger.set_verbosity(0)
       
    def _construct_dynamic_transitions(self):
        """
        Constructs a dok_matrix for storing and updating probabilities
        """

        # Define matrix dimensions
        n_states = len(self.states)
        n_actions = self.max_action_dim
        
        # Initialize list of dok_matrices for transition probabilities and rewards
        dynamic_probs = {state: dok_matrix((n_actions, n_states)) for state in self.encoded_states}
        dynamic_rewards = {state: dok_matrix((n_actions, n_states)) for state in self.encoded_states}
        transition_counts = {state: dok_matrix((n_actions, n_states)) for state in self.encoded_states}
        state_action_counts = dok_matrix((n_states, n_actions))

        # Parse transitions into dok_matrix (if provided)
        if self.raw_transitions != None:
            for (state, action, next_state, probability, reward) in self.raw_transitions:
                # Retrieve state, action, and next_state indices
                encoded_state = self.encoder.encode_state(state)
                encoded_action = self.encoder.encode_action(action)
                encoded_next_state = self.encoder.encode_state(next_state)

                # Assign probability and rewards values
                dynamic_probs[encoded_state][encoded_action, encoded_next_state] = probability
                dynamic_rewards[encoded_state][encoded_action, encoded_next_state] = reward
        
        # Package data into dictionary
        dynamic_transitions = {'probability': dynamic_probs,
                               'rewards': dynamic_rewards,
                               'k': transition_counts,
                               'k_sa': state_action_counts}

        return dynamic_transitions

    def update_transitions(self, params, verbosity = 0):

        # Set method verbosity
        old_verbosity = self._logger.verbosity
        self._logger.set_verbosity(verbosity)
        
        # Invalidate cached probabilities and rewards
        self.cached_probabilities = {}
        self.cached_rewards = {}

        for state, action, next_state, reward in params:
            # Get transition indices
            state_idx = self.encoder.encode_state(state)
            action_idx = self.encoder.encode_action(action)
            next_state_idx = self.encoder.encode_state(next_state)
            ## self._logger(f"NEXT_ID: {next_state}")
            ## self._logger(f"NEXT_STATE_ID: {next_state_idx}")
            # Update transition counts
            self.dynamic_transitions['k'][state_idx][action_idx, next_state_idx] += 1
            self.dynamic_transitions['k_sa'][state_idx, action_idx] += 1

            # Get new transition and action counts
            k = self.dynamic_transitions['k'][state_idx][action_idx, next_state_idx]
            k_sa = self.dynamic_transitions['k_sa'][state_idx, action_idx]
            
            # Update rewards with incremental averaging
            old_val = self.dynamic_transitions['rewards'][state_idx][action_idx, next_state_idx]
            exp_reward = old_val + (reward - old_val)/k
            self.dynamic_transitions['rewards'
                                   ][state_idx
                                   ][action_idx,
                                     next_state_idx] = exp_reward

            # Update probabilities with new transition count
            k_vec = self.dynamic_transitions['k'][state_idx][action_idx, :]
            prob_vec = k_vec/k_sa
            self.dynamic_transitions['probability'
                                    ][state_idx
                                    ][action_idx, :] = prob_vec

            if state != next_state:
                self._logger("--------- Updating the following transition: ----------")
                self._logger(f"Action_idx {action_idx}")
                self._logger(f"State {state_idx}")
                self._logger(f"Action: {action}")
                self._logger(f"Next_State: {next_state_idx}")
                self._logger(f"k_vec/ksa: {prob_vec}")
                self._logger(f"Reward: {exp_reward}")

            # Reset verbosity
            self._logger.set_verbosity(old_verbosity)

    def get_probability_tensor(self, encoded_state):
        if encoded_state not in self.cached_probabilities:
            dok = self.dynamic_transitions['probability'][encoded_state]
            self.cached_probabilities[encoded_state] = dok_to_sparse_tensor(dok)
        return self.cached_probabilities[encoded_state]

    def get_rewards_tensor(self, encoded_state):
        if encoded_state not in self.cached_rewards:
            dok = self.dynamic_transitions['rewards'][encoded_state]
            self.cached_rewards[encoded_state] = dok_to_sparse_tensor(dok)
        return self.cached_rewards[encoded_state]

class EnvChild(EnvModel):
    
    def __init__(self, encoder, actions):

        self.encoder = encoder
        self.actions = actions

