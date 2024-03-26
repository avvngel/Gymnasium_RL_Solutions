#!/usr/bin/env python3

import numpy as np
from env_model import *
from grid_world import GridWorldEnv
from policy import *

#################### Set up Environment ###################

# Create a 4x4 GridWorld Environment
env = GridWorldEnv(4)

# Create environment model
env_model = EnvModel(env.states, 
                     env.get_available_actions,
                     env.transitions)

################# Policy Iteration ##################

# Initialize a random policy to start
pi = get_random_policy(env_model)

# Initialize state-value function to start
V_k = torch.zeros((len(env_model.states), 1))

# Training Params
max_Iters = 5

for i in range(max_Iters):

    print(f"k={i+1}")

    # Cache old policy
    old_pi = pi

    # Find state-value function corresponding to policy pi
    V_k = iterative_evaluation(V_k, pi, env_model, gamma=.99)

    # Improve policy based on state-value function
    pi = improve_policy(pi, env_model, V_k, gamma = .99)

    print(V_k)

################### Simulate Environment ###################

# Simulation Params
#n_episodes
#n_ti

