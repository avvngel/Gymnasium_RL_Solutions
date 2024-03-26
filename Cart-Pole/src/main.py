#!/usr/bin/env python3

# Import necessary modules
import gymnasium as gym
import numpy as np
from env_model import *
from policy import *
from utils import *
from itertools import product
import time

def opp_action(action):
    if action == 1:
        return 0
    elif action == 0:
        return 1

###################### Set up environment ######################

# Create gym environment
env = gym.make('CartPole-v1')#, render_mode = 'human')

# Retrive environment bounds and episode termination bounds
global_bounds = np.concatenate([env.observation_space.low,
                                env.observation_space.high]).reshape(2, 4)
episode_bounds = global_bounds/2

# Discretize state variables
N = 2 # Discretization points in termination window
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
print("-------disc states----------")
print(len(discretized_states))

# Write function mapping states to available actions
def available_actions(state):
    # From gym Documentation:
    # 0: Push cart to the left
    # 1: Push cart to the right
    return [0, 1]

# Create discretized environment model
env_model = EnvModel(discretized_states, available_actions)

# Cache number of states for later
n_states = len(env_model.states)

# Generate random policy to start
pi = get_random_policy(env_model)

##################### Explore Environment #####################

# Loop params
n_episodes = 1000
max_time_steps = 500
fps = 30 # render frames per second

for episode in range(n_episodes):
    env_state = env.reset()[0]
    #env.render()
    for t in range(max_time_steps):
        model_state = get_discretized_state(env_state, discretized_state_vars)
        action = select_action(env_model, model_state, pi)
        next_env_state, reward, terminated, truncated, info = env.step(action)
        next_model_state = get_discretized_state(env_state, discretized_state_vars)

#        time.sleep(1/fps)
        if truncated or terminated:
            reward = 0
            env_model.update_transitions([[model_state, action, next_model_state, reward]])
 #           time.sleep(1/fps)
            break

        env_model.update_transitions([[model_state, action, next_model_state, reward]])


#env.close()

######################## Policy Iteration #####################

# Initizlize state-value function to 0
V_k = torch.zeros((n_states, 1))
max_Iters = 3
epochs = 5
print("k=0")
print(V_k.count_nonzero())
for j in range(epochs):
    print(f"#################### EPOCH {j} #########################")
    # ---------------------Explore environment---------------------
    # Loop params
    n_episodes = 1000
    max_time_steps = 500
    fps = 30 # render frames per second

    for episode in range(n_episodes):
        env_state = env.reset()[0]
        #env.render()
        for t in range(max_time_steps):
            model_state = get_discretized_state(env_state, discretized_state_vars)
            action = select_action(env_model, model_state, pi)
            next_env_state, reward, terminated, truncated, info = env.step(action)
            next_model_state = get_discretized_state(env_state, discretized_state_vars)

    #        time.sleep(1/fps)
            if truncated or terminated:
                reward = -50
                env_model.update_transitions([[model_state, action, next_model_state, reward]])
     #           time.sleep(1/fps)
                break

            env_model.update_transitions([[model_state, action, next_model_state, reward]])


    # --------------------- Policy Iteration ----------------------
    for i in range(max_Iters):
        # Get state-value function for policy
        print("Beginning Policy Evaluation")
        tic = time.time()
        V_k = iterative_evaluation(V_k, pi, env_model, gamma = .8)
        toc = time.time()
        print(f"k = {i+1}")
        non_zero_indices = V_k.nonzero()
        print(V_k[non_zero_indices[:,0], non_zero_indices[:,1]])

        print(f"Policy Evaluation: {toc - tic}s")

        # Policy Improvement
        old_pi = pi
        print("Beginning Policy Improvement")
        tic = time.time()
        pi = improve_policy(pi, env_model, V_k, gamma = .99)
        toc = time.time()

        print(f"Policy Improvement: {toc - tic}s")
        #if torch.all(torch.isclose(old_pi, pi, rtol = 1e-7, atol = 1e-9)):
         #   break

################### Try out new policy ##################
env = gym.make('CartPole-v1', render_mode = 'human')

# Loop params
n_episodes = 1000
max_time_steps = 500
fps = 30 # render frames per second

print("------------- Policy -------------")
#print(pi[997])
#print(list(env_model.dynamic_transitions.keys())[997])

for episode in range(n_episodes):
    env_state = env.reset()[0]
    env.render()
    for t in range(max_time_steps):
        model_state = get_discretized_state(env_state, discretized_state_vars)
        action = select_action(env_model, model_state, pi)
        next_env_state, reward, terminated, truncated, info = env.step(action)
        next_model_state = get_discretized_state(env_state, discretized_state_vars)

        time.sleep(1/fps)
        if truncated or terminated:
            reward = 0
            env_model.update_transitions([[model_state, action, next_model_state, reward]])
            time.sleep(1/fps)
            break

        env_model.update_transitions([[model_state, action, next_model_state, reward]])

