#!/usr/bin/env python3
from typing import Dict, List, Tuple, Set, Union
from scipy.sparse import coo_matrix
from env_model import EnvModel
import numpy as np
import torch
from utils import dok_to_sparse_tensor
import time

def iterative_evaluation(V_k: torch.Tensor,
                          pi: torch.Tensor,
                          env_model: EnvModel,
                          gamma = .99,
                          tol = 1e-6):

    """
    This function executes iterative policy evaluation to obtain the value
    function for the provided policy pi.
    """

    # Form R^pi vector and P^pi matrix
    R_pi, P_pi = form_R_pi_and_P_pi(pi, env_model)

    delta = float('inf') # Initialize difference
    iters = 0
    while delta >= tol:
        v_k = V_k.clone() # Cache current value function
        V_k = R_pi + gamma*torch.matmul(P_pi, V_k) # Update V_k
        change = torch.abs(V_k - v_k).max() # Compute maximum change across all states
        delta = change.item() # Update difference
        iters += 1
    return V_k

def improve_policy(pi, env_model, V_k, gamma = .99):
    
    for state_idx in env_model.encoded_states:
    
        # Get transition probabilities and rewards:
        prob_dist = dok_to_sparse_tensor(env_model.dynamic_transitions['probability'][state_idx])
        rewards = dok_to_sparse_tensor(env_model.dynamic_transitions['rewards'][state_idx])
        
        non_zero_indices = prob_dist.to_dense().nonzero()
        prob_dist_nonzero = prob_dist.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        if prob_dist_nonzero.shape[0] != 0:
            print('------- prob_dist_nonzero -------')
            print(prob_dist_nonzero)
        
        non_zero_indices = rewards.to_dense().nonzero()
        rewards_nonzero = rewards.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        if rewards_nonzero.shape[0] != 0:
            print('-------- rewards_nonzero -------')
            print(rewards_nonzero)
        # Element-wise multiplication of transition probabilities and rewards
        prob_reward_product = prob_dist * rewards

        # Take row sum to sum across next_states
        prob_reward_sumproduct = prob_reward_product.sum(dim = 1).unsqueeze(dim = 1)

        # Product of transition probabitlities and discounted next_state values
        gamma_P_V = gamma*torch.matmul(prob_dist, V_k)
    
        # Compute Q-values
        Q_values = gamma_P_V + prob_reward_sumproduct

        # Select action that maximizes reward if policy pi is followed after
        best_action = torch.argmax(Q_values)

        # Update policy to be greedy w.r.t the current value function
        pi[state_idx, :] = torch.zeros(env_model.max_action_dim)
        pi[state_idx, best_action] = 1
        non_zero_indices = Q_values.nonzero()
        Q_nonzero = Q_values[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        if Q_nonzero.shape[0] != 0:
            print('----- Q-Values nonzero ------')
            print(Q_values[non_zero_indices[:, 0], non_zero_indices[:, 1]])
            print('----------- pi row -----------')
            print(pi[state_idx, :])
            time.sleep(.5)

    return pi


def form_R_pi_and_P_pi(pi, env_model):
    """
    Returns the matrices R^pi and P^pi that are to be used in the iterative
    policy evaluation.
    """
    n_states = len(env_model.states)
    P_pi_row_indices = []
    P_pi_col_indices = []
    P_pi_values = []
    R_pi = torch.zeros((n_states, 1))

    for state in env_model.encoded_states:
        n_actions = env_model.max_action_dim
        pi_row_vec = pi[state].unsqueeze(0)
        prob_dist = dok_to_sparse_tensor(env_model.dynamic_transitions['probability'][state])
        exp_rewards = dok_to_sparse_tensor(env_model.dynamic_transitions['rewards'][state])
        
        # Element-wise multiplication of probabilities and rewards
        prob_reward_product = prob_dist * exp_rewards
        
        # Sum over actions for expected rewards
        expected_rewards = torch.matmul(pi_row_vec, prob_reward_product).sum(dim = 1)
        R_pi[state] = expected_rewards

        # Compute each row of P_pi
        P_pi_row = torch.matmul(pi_row_vec, prob_dist).squeeze()

        # Adjust row indices and store COO information
        non_zero_indices = P_pi_row.nonzero(as_tuple = True)[0].tolist()
        adjusted_row_indices = [state]*len(non_zero_indices)
        P_pi_row_indices.extend(adjusted_row_indices)
        P_pi_col_indices.extend(non_zero_indices)
        P_pi_values.extend(P_pi_row[non_zero_indices].tolist())
        
         
    # Combine P_pi_rows into one sparse_coo_tensor
    indices = torch.tensor([P_pi_row_indices, P_pi_col_indices], dtype = torch.long)
    values = torch.tensor(P_pi_values, dtype = torch.float32)
    P_pi = torch.sparse_coo_tensor(indices, values, (n_states, n_states))
    return R_pi, P_pi
    

def get_random_policy(env_model):

    """
    Returns a random policy for a given Env_Model.
    """
    
    # Define matrix dims
    n_states = len(env_model.states)
    n_actions = env_model.max_action_dim

    # Instantiate lil_matrix
    random_policy = torch.zeros((n_states, n_actions), dtype = torch.float32)

    for state in env_model.encoded_states:
        n_available_actions = len(env_model.actions(state))
        probability = 1/n_available_actions
        random_policy[state, :n_available_actions] = probability

    return random_policy

def get_empty_policy(EnvModel):

    pass


