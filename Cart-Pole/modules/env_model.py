#!/usr/bin/env python3

from utils import Encoder, dok_to_sparse_tensor
from scipy.sparse import lil_matrix, dok_matrix 
import torch

class EnvModel():

    def __init__(self, states, actions, transitions = None):
        self.states = states
        self.actions = actions
        self.max_action_dim = max([len(actions(state)) for state in self.states])
        self.raw_transitions = transitions
        self.encoder = Encoder()
        self.encoded_states = self.encoder.encode_states(states)
        self.dynamic_transitions = self._construct_dynamic_transitions()
       
    def _construct_dynamic_transitions(self):
        """
        Constructs a dok_matrix for storing and updating probabilities
        """

        # Define matrix dimensions
        n_states = len(self.states)
        n_actions = self.max_action_dim
        
        # Initialize list of dok_matrices for transition probabilities and rewards
        dynamic_probs = [dok_matrix((n_actions, n_states)) for state in self.states]
        dynamic_rewards = [dok_matrix((n_actions, n_states)) for state in self.states]
        transition_counts = [dok_matrix((n_actions, n_states)) for state in self.states]
        state_action_counts = dok_matrix((n_states, n_actions))

        # Parse transitions into dok_matrix (if provided)
        if self.raw_transitions != None:
            for (state, action, next_state, probability, reward) in self.raw_transitions:
                # Retrieve state, action, and next_state indices
                encoded_state = self.encoder.encode_state(state)
                encoded_action = self.encoder.encode_action(encoded_state, action)
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

    def coo_transitions(self):
        """
        This function converts the dynamic dok_matrix into a coo_sparse_tensor
        for computation and returns it.
        """
        probs = []
        rewards = []
        for state in self.encoded_states:
            probs.append(dok_to_sparse_tensor(self.dynamic_transitions['probability'][state]))
            rewards.append(dok_to_sparse_tensor(self.dynamic_transitions['rewards'][state]))

        return{'probability': probs,
               'rewards': rewards}

    def update_transitions(self, params):
        for state, action, next_state, reward in params:
            # Get transition indices
            state_idx = self.encoder.encode_state(state)
            action_idx = self.encoder.encode_action(state_idx, action)
            next_state_idx = self.encoder.encode_state(next_state)
            
            # Update transition counts
            self.dynamic_transitions['k'][state_idx][action_idx, next_state_idx] += 1
            self.dynamic_transitions['k_sa'][state_idx, action_idx] += 1

            # Get new transition and action counts
            k = self.dynamic_transitions['k'][state_idx][action_idx, next_state_idx]
            k_sa = self.dynamic_transitions['k_sa'][state_idx, action_idx]
    
            # Update probabilities with new transition count
            self.dynamic_transitions['probability'
                                   ][state_idx
                                   ][action_idx,
                                     next_state_idx] = k/k_sa

            # Update rewards with incremental averaging
            old_val = self.dynamic_transitions['probability'][state_idx][action_idx, next_state_idx]
            self.dynamic_transitions['rewards'
                                   ][state_idx
                                   ][action_idx,
                                     next_state_idx] = old_val + (reward - old_val)/k
