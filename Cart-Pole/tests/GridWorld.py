#!/usr/bin/env python3

import numpy as np
from env_model import *
from grid_world import GridWorldEnv
from policy import *
from log_controller import *
import argparse
import faulthandler
import numba

#numba.gdb_init()

faulthandler.enable()

# Accept command line arguments
parser = argparse.ArgumentParser(prog='CartPole',
                                description='Process some integers.')
parser.add_argument('--n', type=int, default = 4,
                    help="Number of discretization points (default: 2)")
args = parser.parse_args()

# Instantiate a Logger for controlling output verbosity
logger = Logger()
logger.set_verbosity(0)

#----------------------- Set up Environment -----------------------#

# Env params
n = args.n

# Create a 4x4 GridWorld Environment
env = GridWorldEnv(n)

# Create environment model
env_model = EnvModel(env.states, 
                     env.get_available_actions,
                     env.transitions)

#------------------------ Policy Iteration ------------------------#

# Initialize a random policy to start
pi = get_random_policy(env_model, dist='dirichlet')

# Initialize state-value function to start
V_k = torch.zeros((len(env_model.states), 1))

# Training Params
max_Iters = 100

for i in range(max_Iters):

    logger(f"k={i+1}")

    # Cache old policy
    old_pi = pi.clone()

    # Find state-value function corresponding to policy pi
    V_k = iterative_evaluation(V_k, pi, env_model, gamma=.99, debug_mode = True)

    # Improve policy based on state-value function
    pi = improve_policy(pi, env_model, V_k, gamma = .99, verbosity=0)

    logger('policy:')
    logger(pi)

    if torch.all(torch.isclose(old_pi, pi, rtol = 1e-12)):
        break

#--------------------------- Display Results ----------------------#
for row in range(n):
    print('+' + '-'*(4*n - 1) + '+')
    print('|', end = '')
    for col in range(n):
        state = row*n + col
        if state in env.terminal_states:
            action = 'x'
        else:
            action = env.action_reprs[torch.argmax(pi[row*n + col]).item()]
        print(f" {action} |", end='')
    print('')
print('+' + '-'*(4*n - 1) + '+')

#------------------------ Simulate Environment --------------------#

# Simulation Params
#n_episodes
#n_ti

