#!/usr/bin/env python3

# Import necessary modules
import gymnasium as gym
import numpy as np
from env_model import *
from policy import *
from utils import *
from itertools import product
import time
from log_controller import *

# Accept command line arguments
parser = argparse.ArgumentParser(prog='CartPole', 
                                description='Process some integers.')
parser.add_argument('--N', type=int, default = 2,
                    help="Number of discretization points (default: 2)")
parser.add_argument('--n_epochs', type=int, default=1,
                    help='Number of training epochs (default: 1)')
parser.add_argument('--demonstrate', type=int, default=0,
                    help='If set to 1, will render a pyGame demsonstration')
parser.add_argument('--epsilon', type=float, default=0,
                    help='If >0, uses as the epsilon for epsilon greedy exploration')
parser.add_argument('--gamma', type=float, default=.8,
                    help='Sets discount factor for reward calculation in policy evaluation')

args = parser.parse_args()

# Create a logger for controlling output verbosity
logger = Logger()
logger.set_verbosity(1)

###################### Set up environment ######################

# Create gym environment
env = gym.make('CartPole-v1')#, render_mode = 'human')

logger("Discretizing environment...", msg_verbosity = 1, end = '')
tic = time.time()

# Retrive environment bounds and episode termination bounds
global_bounds = np.concatenate([env.observation_space.low,
                                env.observation_space.high]).reshape(2, 4)
episode_bounds = global_bounds/2

# Discretize state variables
N = args.N # Discretization points in termination window
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

logger("Initializing EnvModel...", msg_verbosity = 1, end = '')
tic = time.time()

# Create discretized environment model
env_model = EnvModel(discretized_states, available_actions)

toc = time.time()
logger(f"{toc - tic:.3f}s", msg_verbosity = 1)

# Cache number of states for later
n_states = len(env_model.states)

logger("Initializing policy...", msg_verbosity = 1, end = '')
tic = time.time()

# Generate random policy to start
pi = get_random_policy(env_model, dist = 'discrete', extra_param = [.5, .5])

toc = time.time()
logger(f"{toc - tic:.3f}s", msg_verbosity = 1)

############################## Policy Iteration ###############################

# Exploration loop params
n_episodes = 1000
max_time_steps = 500
epsilon = args.epsilon # Percentage of time to go against greedy policy

gamma = args.gamma

# Initizlize state-value function to 0
V_k = torch.zeros((n_states, 1))

# ********************* Explore environment **********************
# Exploration loop
logger("Exploring environment...", msg_verbosity = 1)
exploration_tic = time.time()
episode_rewards = np.zeros(n_episodes)

for episode in range(n_episodes):

    logger('')
    logger('')
    logger(f"#************************* EPISODE {episode + 1} ******************************#")

    # Reset environment to initial state
    env_state = env.reset()[0]

    for t in range(max_time_steps):

        # Convert continuous environment state into discrete model state
        model_state = get_discretized_state(env_state, discretized_state_vars)

        # Select action based on policy
        action = select_action(env_model, model_state, pi, epsilon = epsilon)

        # Execute action and record environment reaction
        env_state, reward, terminated, truncated, info = env.step(action)

        # Convert continuous next environment state in to discrete next model state
        next_model_state = get_discretized_state(env_state, discretized_state_vars)
            
        # Debugging output
        logger(f"Discretized State: {env_model.encoder.encode_state(model_state)}")
        logger(f"Action taken: {action}")
        logger(f"Discretized Next State: {env_model.encoder.encode_state(next_model_state)}")
            
        # Check for episode termination or truncation
        if truncated or terminated:

            reward = -2
            env_model.update_transitions([[model_state, action, next_model_state, reward]])
            break
            
        # record_rewards
        episode_rewards[episode] += 1

        # Update transition probabilities and expected rewards
        env_model.update_transitions([[model_state, action, next_model_state, reward]])
            
        # Debuging output
        state_idx = env_model.encoder.encode_state(model_state)
        next_state_idx = env_model.encoder.encode_state(next_model_state)
        action_idx = env_model.encoder.encode_action(action)
        logger(f"Discretized State: {state_idx}")
        logger(f"Action taken: {action}")
        logger(f"Discretized Next State: {next_state_idx}")
        logger(f"Probability: {env_model.dynamic_transitions['probability'][state_idx][action_idx, next_state_idx]}")
        logger(f"Reward as indexed: {env_model.dynamic_transitions['rewards'][state_idx][action_idx, next_state_idx]}")
        logger(f"Reward as inserted: {reward}")
    
exploration_toc = time.time()
logger(f"Exploration time: {exploration_toc - exploration_tic:.3f}s", msg_verbosity = 1)

# *********************** Policy Iteration ***********************
logger("Beginning policy iteration...", msg_verbosity = 1)
policy_iteration_tic = time.time()

for i in range(max_Iters):
        
    # Display policy iteration
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

    # Get policy that's greedy to V_k
    pi = improve_policy(pi, env_model, V_k, gamma = .99)
    improve_policy_toc = time.time()
    logger(f"{improve_policy_toc - improve_policy_tic:.2f}")
