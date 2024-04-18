#!/usr/bin/env python3

import numpy as np
from env_model import *
from grid_world import GridWorldEnv
from policy import *
from log_controller import *
from utils import select_action
import argparse
import pdb


# Accept command line arguments
parser = argparse.ArgumentParser(prog='GridWorld',
                                description='Process some integers.')
parser.add_argument('--n', type=int, default = 4,
                    help="Number of discretization points (default: 2)")
args = parser.parse_args()

# Instantiate a Logger for controlling output verbosity
logger = Logger()
logger.set_verbosity(2)

#----------------------- Set up Environment -----------------------#

# Env params
n = args.n

# Create a 4x4 GridWorld Environment
env = GridWorldEnv(n)

# Create environment model
env_model = EnvModel(env.states, 
                     env.get_available_actions)

# Initialize a random policy to start
pi = get_random_policy(env_model, dist='dirichlet')

#************************** Training Loop **************************#
# Params
n_epochs = 10

for _ in range(n_epochs):
    #----------------------- Explore Environment ----------------------#

    # Exploration params
    n_episodes = 100
    max_time_steps = 500
    params = []
    episode_rewards = np.zeros(n_episodes)

    #pdb.set_trace()
    print("Beginning Exploration...", end = '')
    expl_tic = time.time()
    for episode in range(n_episodes):
        
        # reset Environment
        state = env.reset()

        for t in range(max_time_steps):

            #logger(f"State: {state}")

            # Select random action
            action = select_action(env_model, state, pi, epsilon = 0)
            
            #logger(f"Action taken: {action}")

            # Execute action and record reward
            next_state, reward, terminated, truncated, info = env.step(action)

            # Debugging output
            #logger(f"Next State: {next_state}")

            # Check for episode termination or truncation
            if terminated:

                reward = 0
                params.append((state, action, next_state, float(reward)))

                # Update transitions
                env_model.update_transitions([(state, action, next_state, float(reward))])

                break

            # record_rewards
            episode_rewards[episode] -= 1

            # Update transition probabilities and expected rewards
            params.append((state, action, next_state, float(reward)))

            # Update transitions
            env_model.update_transitions([(state, action, next_state, float(reward))], verbosity = 2)

            # Update state
            state = next_state

    expl_toc = time.time()

    print(f"{expl_toc - expl_tic:.3f}s")


    #------------------------ Policy Iteration ------------------------#

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

