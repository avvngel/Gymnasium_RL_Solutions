#!/usr/bin/env python3

# Import necessary modules
import gymnasium as gym
import numpy as np
from env_model import *
from policy import *
from utils import *
from itertools import product
import time
import argparse
import multiprocessing as mp
from log_controller import *
import os
import torch
import random
import pdb

if __name__ == "__main__":

    # Accept command line arguments
    parser = argparse.ArgumentParser(prog='CartPole', 
                                    description='Process some integers.')
    
    parser.add_argument('--N', type=int, default = 2,
                        help="Number of discretization points (default: 2)")
    
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')

    parser.add_argument('--n_training_episodes', type=int, default=10000,
                        help='Number of training episodes per epoch (default: 10,000)')
    
    parser.add_argument('--demonstrate', type=int, default=0,
                        help='If set to 1, will render a pyGame demsonstration after training.')
    
    parser.add_argument('--epsilon', type=float, default=0,
                        help='If >0, uses as the epsilon for epsilon greedy exploration')
    
    parser.add_argument('--min_epsilon', type=float, default=0,
                        help='Sets minimum value for epsilon for decaying epsilon-greedy exploration')
    
    parser.add_argument('--gamma', type=float, default=.8,
                        help='Sets discount factor for reward calculation in policy evaluation')
    
    parser.add_argument('--epsilon_decay_rate', type=float, default=.02,
                        help='Sets the decay rate per epoch of epsilon used in epsilon-greedy exploration.')

    parser.add_argument('--env_name', type=str, default="CartPole-v1",
                        help='The name of the Gymnasium environment to solve policy for.')

    parser.add_argument('--seed', type=int, default=None,
                        help='Sets Random seed for gym environment')
    
    parser.add_argument('--n_cores', type=int, default=1,
                        help='Number of course for distributed inference parallel exploration.')

    parser.add_argument('--v', type=int, default=1,
                        help='Set verbosity for output.')

    # Parse command line arguments
    args = parser.parse_args()

    N                  = args.N
    n_epochs           = args.n_epochs
    n_episodes         = args.n_training_episodes
    epsilon            = args.epsilon
    min_epsilon        = args.min_epsilon
    gamma              = args.gamma
    epsilon_decay_rate = args.epsilon_decay_rate
    seed               = args.seed
    n_tasks            = args.n_cores
    verbosity          = args.v
    demonstrate        = args.demonstrate
    env_name           = args.env_name

    # Set global seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    #seed_everything(seed)

    # Create a logger for controlling output verbosity
    logger = Logger()
    logger.set_verbosity(verbosity)

    ###################### Set up environment ######################

    # Create gym environment
    env = gym.make(env_name)

    # Print checkpoint message and start discretization timer
    logger("Discretizing environment...", msg_verbosity = 1, end = '')
    tic = time.time()

    # Retrive environment bounds and episode termination bounds
    global_bounds = np.concatenate([env.observation_space.low,
                                    env.observation_space.high]).reshape(2, 4)
    episode_bounds = global_bounds/2

    # Discretize state variables
    n_state_vars = env.observation_space.shape[0]
    discretized_state_vars = []

    for state_var_idx in range(n_state_vars):

        # Get episode termination bounds
        lb = episode_bounds[:, state_var_idx][0]
        ub = episode_bounds[:, state_var_idx][1]


        # Set bound based for infinite termination windows
        if is_unbounded([lb, ub]):
            lb = -10
            ub = 10
         
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

    # Discretize states
    discretized_states = list(product(*discretized_state_vars))

    toc = time.time()
    logger(f"{toc - tic:.3f}s", msg_verbosity = 1)

    # Write function mapping states to available actions
    def available_actions(state):
        """
        Returns the available actions for the provided state according to 
        the Gymnasium Cart-Pole v2 specifications.
        """
        # From gym Documentation:
        # 0: Push cart to the left
        # 1: Push cart to the right
        return [0, 1]

    # Print Checkpoint message and start initialization timer
    logger("Initializing EnvModel...", msg_verbosity = 1, end = '')
    tic = time.time()

    # Create discretized environment model
    env_model = EnvModel(discretized_states, available_actions)

    # Log initialization time
    toc = time.time()
    logger(f"{toc - tic:.3f}s", msg_verbosity = 1)

    # Cache number of states for later
    n_states = len(env_model.states)

    # Print checkpoint message and start policy initialization timer
    logger("Initializing policy...", msg_verbosity = 1, end = '')
    tic = time.time()

    # Generate random policy to start
    pi = get_random_policy(env_model, dist = 'discrete', extra_param = [.5, .5])

    # Log policy generation time
    toc = time.time()
    logger(f"{toc - tic:.3f}s", msg_verbosity = 1)

    ############################## Policy Iteration ###############################

    # Exploration loop params
    max_time_steps = 500
    fps = 30 # render frames per second
    epsilon += epsilon_decay_rate
    partition = [int(n_episodes / n_tasks)] * n_tasks

    # Distribute any remaining episodes to the last parition
    partition[-1] += n_episodes % n_tasks

    # Policy iteration loop params
    max_Iters = 1000

    logger("Beginning training loop...", msg_verbosity = 1)
    total_time_tic = time.time()
    logger("k=0")
   
    # Training Loop
    for j in range(n_epochs):

        logger(f"#################### EPOCH {j+1} #########################", msg_verbosity = 1)
        epoch_tic = time.time()

        # Decay epsilon as epochs go on
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        logger(f"EPSILON: {epsilon}", msg_verbosity = 1)

        # ********************* Explore environment **********************
        # Exploration loop
        logger("Exploring environment...", msg_verbosity = 1)
        exploration_tic = time.time()
        
        # Instantiate Queue to receive exploration results
        results_queue = mp.Queue()

        # Instantiate all processes
        processes = [mp.Process(target=explore, args=(env, env_model.child(), pi,
                                                      epsilon, discretized_state_vars,
                                                       partition[i], max_time_steps,
                                                       results_queue, seed)
                                )
                     for i in range(n_tasks)]

        # Run all processes
        for p in processes:
            p.start()

        results_received = 0

        while results_received < n_tasks:

            result = results_queue.get()
            results_received += 1
            
            # Update transition probabilities and expected rewards
            tic = time.time()
            env_model.update_transitions(result, verbosity = 0)
            toc = time.time()
            logger(f"Processed result {results_received} in {toc - tic:.3f}s", 
                    msg_verbosity = 1)

            # Prevent busy waiting 
        for p in processes:
            p.join()
        
        exploration_toc = time.time()
        logger(f"Exploration time: {exploration_toc - exploration_tic:.3f}s", 
                msg_verbosity = 1)

        # *********************** Policy Iteration ***********************
        logger("Beginning policy iteration...", msg_verbosity = 1)
        policy_iteration_tic = time.time()

        # Initizlize state-value function to 0
        V_k = torch.zeros((n_states, 1))

        for i in range(max_Iters):
            
            # Display policy iteration #
            logger(f"k = {i+1}", msg_verbosity = 1)

            # ---------- Policy Evaluation -----------
            logger("Policy Evaluation: ", msg_verbosity = 1, end = '')
            
            # Get state-value function for policy and time execution
            policy_eval_tic = time.time()
            V_k = iterative_evaluation(V_k, pi, env_model, gamma = gamma)
            policy_eval_toc = time.time()
            
            # Display nonzero elements of policy
            non_zero_indices = V_k.nonzero()
            logger(V_k[non_zero_indices[:,0], non_zero_indices[:,1]])
            logger(f"{policy_eval_toc - policy_eval_tic:.3f}s", msg_verbosity = 1)

            # ---------- Policy Improvement ----------
            logger("Policy Improvement: ", msg_verbosity = 1, end = '')
            improve_policy_tic = time.time()

            # Cache old policy
            old_pi = pi.clone()

            # Get policy that's greedy to V_k
            pi = improve_policy(pi, env_model, V_k, gamma = .99, verbosity = 0)
            improve_policy_toc = time.time()

            # Display execution time
            logger(f"{improve_policy_toc - improve_policy_tic:.3f}s", msg_verbosity = 1)

            # Display non-zero policy entries
            logger("------------- Policy -------------", msg_verbosity = 2)
            non_zero_indices = pi.nonzero()
            logger(non_zero_indices, msg_verbosity = 2)
            logger(pi[non_zero_indices[:,0], :], msg_verbosity = 2)
            logger(env_model.encoder.action_encoding, msg_verbosity = 2)

            # Exit criterion
            if torch.all(torch.isclose(old_pi, pi, rtol = 1e-7, atol = 1e-7)):
                policy_iteration_toc = time.time()
                logger(f"Policy iteration terminated in {i+1} iters and {policy_iteration_toc - policy_iteration_tic:.3f}s.", msg_verbosity = 1)
                break
        epoch_toc = time.time()
        logger(f"Epoch {j+1} time: {epoch_toc - epoch_tic:.3f}s", msg_verbosity = 1)

    total_time_toc = time.time()
    logger(f"Training finished in {total_time_toc - total_time_tic:.3f}s", msg_verbosity = 1)

    # Save new policy
    policy_name = f"CartPole_policy_N_{N}_{n_epochs}_epochs_{epsilon}_epsilon_{gamma}_gamma_{epsilon_decay_rate}_decay_rate_seed_{seed}.pt"
    torch.save(pi, policy_name)

    # Display non-zero policy entries
    #logger("------------- Policy -------------")
    #non_zero_indices = pi.nonzero()
    #logger(pi[non_zero_indices[:,0], :])
    #logger(env_model.encoder.action_encoding)

    ######################## Launch Policy Demonstration ###########################

    # Create new environment
    if demonstrate:
        env = gym.make(env_name, render_mode = 'human')

    else:
        env = gym.make(env_name)


    # Loop params
    n_episodes = 100
    max_time_steps = 500
    fps = 30 # render frames per second

    # Pre-allocate reward tally for computing average reward per episode
    reward_tally = np.zeros(n_episodes)

    # Demonstration loop
    for episode in range(n_episodes):

        # Reset environment to initial state
        if seed == None:
            env_state = env.reset()[0]
        else:
            env_state = env.reset(seed=seed)[0]
        
        if demonstrate:
            # Render Pygame animation
            env.render()

        for t in range(max_time_steps):

            # Convert continuous environment state into discrete model state 
            model_state = get_discretized_state(env_state, discretized_state_vars)

            # Select action according to policy
            action = select_action(env_model, model_state, pi)

            # Execute action and record environment reaction
            env_state, reward, terminated, truncated, info = env.step(action)

            # Convert continuous next environment state in to discrete next model state
            next_model_state = get_discretized_state(env_state, discretized_state_vars)

            # Record reward
            reward_tally[episode] += reward
            
            if demonstrate:

                # Sleep to smoothen out animation
                time.sleep(1/fps)
            
            # Check exit criterion
            if truncated or terminated:
                break

    print(f"Average reward: {reward_tally.mean()}")

