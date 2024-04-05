#!/usr/bin/env python3
from typing import Dict, List, Tuple, Set, Union
from scipy.sparse import coo_matrix
from env_model import EnvModel
from torch.distributions import Dirichlet
import numpy as np
import torch
from utils import dok_to_sparse_tensor
import time
from log_controller import *
import argparse

# Instantiate Logger to control output verbosity
logger = Logger()
logger.set_verbosity(0)

def iterative_evaluation(V_k: torch.Tensor,
                          pi: torch.Tensor,
                          env_model: EnvModel,
                          gamma = .99,
                          tol = 1e-6,
                          debug_mode = False):

    """
    This function executes iterative policy evaluation to obtain the value
    function for the provided policy pi.
    """

    # Set verbosity
    verbose_output = debug_mode

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
    
    # Reset Verbosity
    verbose_output = False

    return V_k

#@profile
def improve_policy(pi, env_model, V_k, gamma = .99, debug_mode = False, verbosity = 0):
    old_verbosity = logger.verbosity
    logger.set_verbosity(verbosity)
    non_zero_Q = []
    for state_idx in env_model.encoded_states:
    
        # Get transition probabilities and rewards:
        prob_dist = env_model.get_probability_tensor(state_idx)
        rewards = env_model.get_rewards_tensor(state_idx)
        
        if verbosity == 2:
            non_zero_indices = prob_dist.to_dense().nonzero()
            prob_dist_nonzero = prob_dist.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            if prob_dist_nonzero.shape[0] != 0:
                logger('------- prob_dist_nonzero -------')
                logger(prob_dist_nonzero)
        
            non_zero_indices = rewards.to_dense().nonzero()
            rewards_nonzero = rewards.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            if rewards_nonzero.shape[0] != 0:
                logger('-------- rewards_nonzero -------')
                logger(rewards_nonzero)
            
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

        if verbosity == 2:
            non_zero_indices = prob_dist.to_dense().nonzero()
            prob_dist_nonzero = prob_dist.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            if prob_dist_nonzero.shape[0] != 0:
                logger('------- prob_dist_nonzero -------')
                logger(prob_dist_nonzero)

            non_zero_indices = rewards.to_dense().nonzero()
            rewards_nonzero = rewards.to_dense()[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            if rewards_nonzero.shape[0] != 0:
                logger('-------- rewards_nonzero -------')
                logger(rewards_nonzero)
            gamma_P_V_nzidx = gamma_P_V.to_dense().nonzero()
            gamma_P_V_nz = gamma_P_V.to_dense()[gamma_P_V_nzidx[:, 0], gamma_P_V_nzidx[:, 1]]
            pr_sp_nzidx = prob_reward_sumproduct.to_dense().nonzero()
            pr_sp_nz = prob_reward_sumproduct.to_dense()[pr_sp_nzidx[:, 0], pr_sp_nzidx[:, 1]]
            zero = torch.zeros((Q_values.shape[0], 1))

            if not torch.all(torch.eq(Q_values, zero)):

                logger('-------- Q_values -----------')
                logger(f'prob_reward_product shape: {prob_reward_product.shape}')
                logger(f'prob_reward_sumproduct shape: {prob_reward_sumproduct.shape}')
                logger(f'prob_dist shape: {prob_dist.shape}')
                logger(f'V_k shape: {V_k.shape}')
                logger(f'gamma_P_V shape: gamma_P_V.shape[]')
                logger(f'gamma_P_V {gamma_P_V}')
                logger(f'prob_dist {prob_dist}')
                logger(f'rewards {rewards}')
                logger(f'prob_reward_product {prob_reward_product}')
                logger(f'prob_reward_sumproduct {prob_reward_sumproduct}')
                logger(f"Q_values: {Q_values}")
                logger(f"State_idx: {state_idx}")
                logger(f"State value: {env_model.encoder.decode_state(state_idx)}")
                non_zero_Q.append(state_idx)

            non_zero_indices = Q_values.nonzero()
            Q_nonzero = Q_values[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            if Q_nonzero.shape[0] != 0:
                logger('----- Q-Values nonzero ------')
                logger(Q_values[non_zero_indices[:, 0], non_zero_indices[:, 1]])
                logger('----------- pi row -----------')
                logger(pi[state_idx, :])
                time.sleep(.5)

    if verbosity == 2:
        print("NON_ZERO_Q:")
        print(non_zero_Q)

    logger.set_verbosity(old_verbosity)
    return pi
    
#@profile
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
        prob_dist = env_model.get_probability_tensor(state)
        exp_rewards = env_model.get_rewards_tensor(state)
        
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
    

def get_random_policy(env_model, dist, extra_param=None, alpha = None):

    """
    Returns a random policy for a given Env_Model.
    """
    
    # Define matrix dims
    n_states = len(env_model.states)
    n_actions = env_model.max_action_dim

    # Initialize zero-tensor
    random_policy = torch.zeros((n_states, n_actions), dtype = torch.float32)
    
    for state in env_model.encoded_states:
        n_available_actions = len(env_model.actions(state))
        if dist == "uniform":
            probability = 1/n_available_actions
            random_policy[state, :n_available_actions] = probability
        elif dist == "dirichlet":
            alpha = torch.rand(n_available_actions)
            weights = Dirichlet(alpha).sample()
            random_policy[state, :n_available_actions] = weights
        elif dist == "discrete":
            random_policy[state, :n_available_actions] = torch.tensor(extra_param)
        else:
            print(f"dist {dist} not yet supported.")

    return random_policy

def get_empty_policy(env_model):

    """
    Returns a random policy for a given Env_Model.
    """
    # Define matrix dims
    n_states = len(env_model.states)
    n_actions = env_model.max_action_dim

    # Initialize zero-tensor
    empty_policy = torch.zeros((n_states, n_actions), dtype = torch.float32)

    return empty_policy

