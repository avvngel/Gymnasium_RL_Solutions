#!/usr/bin/env python3

# Import necessary packages
import numpy as np
import time
from utils import *
from env_model import *
from policy import *
from log_controller import *
import gymnasium as gym

if __name__ == "__main__":

    # Initialize Logger for controlling output verbosity
    logger = Logger()

    # ------------------------------ Process Command Line Arguments -------------------------------
    
    # Accept command line arguments
    parser = argparse.ArgumentParser(prog='CartPole',
                                    description='Process some integers.')

    parser.add_argument('--N', type=int, default = 2,
                        help="Number of discretization points (default: 2)")

    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')

    parser.add_argument('--n_training_episodes', type=int, default=10000,
                        help='Number of training episodes (default: 10,000)')

    parser.add_argument('--max_time_steps', type=int, default=500,
                        help='Number of time steps per episode (default: 500)')

    parser.add_argument('--epsilon', type=float, default=0,
                        help='If >0, uses as the epsilon for epsilon greedy exploration')

    parser.add_argument('--gamma', type=float, default=.8,
                        help='Sets discount factor for reward calculation in policy evaluation')

    parser.add_argument('--epsilon_decay_rate', type=float, default=.02,
                        help='Sets the decay rate per epoch of epsilon used in epsilon-greedy exploration.')

    parser.add_argument('--n_cores', type=int, default=1,
                        help='Number of course for distributed inference parallel exploration.')

    parser.add_argument('--v', type=int, default=1,
                        help='Set verbosity for output.')

    parser.add_argument('--demonstrate', type=int, default=0,
                        help='If set to 1, will render a pyGame demsonstration')

    parser.add_argument('--env_name', type=str, default="CartPole-v1",
                        help='The name of the Gymnasium environment to solve policy for.')

    parser.add_argument('--seed', type=int, default=None,
                        help='The name of the Gymnasium environment to solve policy for.') 

    # Parse command line arguments
    args = parser.parse_args()

    N                  = args.N
    n_epochs           = args.n_epochs
    n_episodes         = args.n_training_episodes
    max_time_steps     = args.max_time_steps
    epsilon            = args.epsilon
    gamma              = args.gamma
    epsilon_decay_rate = args.epsilon_decay_rate
    seed               = args.seed
    n_cores            = args.n_cores
    verbosity          = args.v
    demonstrate        = args.demonstrate
    env_name           = args.env_name
    
    # Set output verbosity based on command line argument
    logger.set_verbosity(verbosity)

    # Display parsed arguments
    display_width = 90
    display_parsed_args(args, display_width, logger, verbosity=1)

    # ---------------------------- Set up Gym Environment and EnvModel ----------------------------
    
    # Initialize Gymnasium CartPole Environment
    logger(f"Loading environment {env_name}...".center(display_width), msg_verbosity = 1)
    logger("-"*display_width, msg_verbosity = 1)
    env = gym.make(env_name)

    # Retrieve env parameters
    global_bounds = np.concatenate([env.observation_space.low,
                                    env.observation_space.high]).reshape(2, 4)
    episode_bounds = global_bounds/2

    # Display env_bounds
    #display_env_info(global_bounds, episode_bounds)
    n_state_vars = global_bounds.shape[1]
    header = ""
    tab = "      "
    column_title = "State Variable "
    col_width = len(column_title)
    for i in range(n_state_vars):
        header += tab + column_title + str(i+1)
    upper_bound_row = "Upper:"
    lower_bound_row = "Lower:"
    for i in range(n_state_vars):
        upper_bound_row += format(global_bounds[1, i], ".2E").center(col_width+1) + tab
        lower_bound_row += format(global_bounds[0, i], ".2E").center(col_width+1) + tab
    logger("Global environment bounds:", msg_verbosity=1)
    logger("", msg_verbosity=1)
    logger(header, msg_verbosity=1)
    logger(upper_bound_row, msg_verbosity=1)
    logger(lower_bound_row, msg_verbosity=1)
    logger("", msg_verbosity=1)
    logger("Episode termination bounds:", msg_verbosity=1)
    logger("", msg_verbosity=1)

    upper_bound_row = "Upper:"
    lower_bound_row = "Lower:"
    for i in range(n_state_vars):
        upper_bound_row += format(episode_bounds[1, i], ".2E").center(col_width+1) + tab
        lower_bound_row += format(episode_bounds[0, i], ".2E").center(col_width+1) + tab
    logger(header, msg_verbosity=1)
    logger(upper_bound_row, msg_verbosity=1)
    logger(lower_bound_row, msg_verbosity=1)
    logger("", msg_verbosity=1)

    # Log checkpoint message and begin discretization timer
    logger("Discretizing Environment...", end = "", msg_verbosity = 1)
    tic = time.time()

    # Get discretized state variables
    discretized_state_vars = discretize_state_variables(global_bounds, episode_bounds, N)
    
    # Get discetized states from cartesian product
    discretized_states = list(product(*discretized_state_vars))
    
    # Log discretization time
    toc = time.time()
    logger(f"{toc - tic:.3f}s".rjust(6, "."), msg_verbosity = 1)

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
    logger(f"{toc - tic:.3f}s".rjust(9, "."), msg_verbosity = 1)

    # Cache number of states for later
    n_states = len(env_model.states)

    # Print checkpoint message and start policy initialization timer
    logger("Initializing policy...", msg_verbosity = 1, end = '')
    tic = time.time()

    # Generate random policy to start
    pi = get_random_policy(env_model, dist = 'dirichlet')#, extra_param = [.5, .5])

    # Log policy generation time
    toc = time.time()
    logger(f"{toc - tic:.3f}s".rjust(11, "."), msg_verbosity = 1)
    
    # ********************************** TRAINING ***********************************

    # Log checkpoint
    logger("-"*display_width, msg_verbosity = 1)
    logger(f"Beginning Training...".center(display_width), msg_verbosity = 1)
    logger("-"*display_width, msg_verbosity = 1)
    logger("", msg_verbosity = 1)
    training_tic = time.time()

    # Initialize set for storing unique states visited
    visited_states = set()
    
    # Training Loop
    for epoch in range(n_epochs):

        # Log epoch #
        logger(f"Epoch {epoch+1}:", msg_verbosity=1)
        logger("", msg_verbosity=1)
        logger(tab + "Exploring environment...", msg_verbosity=1)
        logger("", msg_verbosity=1)

        # --------------------- Explore Environment ---------------------
        
        # Initialize set for storing unique states visited during this exploration phase
        states_visited_this_epoch = set()

        # Episodic Loop
        for episode in range(n_episodes):

            # Log episode #
            logger(tab + f"Episode {episode + 1}:")

            # Reset environment based on seed and store initial state
            if seed == None:
                env_state = env.reset()[0]
            else:
                env_state = env.reset(seed=seed)[0]

            for t in range(max_time_steps):

                # Log time step #
                logger(tab + f"t = {t+1}")

                # Get model state based on state variable discretization
                model_state = get_discretized_state(env_state, discretized_state_vars)

                # Add state encoding to set of visited states
                visited_states.add(env_model.encoder.encode_state(model_state))
                states_visited_this_epoch.add(env_model.encoder.encode_state(model_state))

                # Select action based on policy
                action = select_action(env_model, model_state, pi, epsilon)

                # Execute action and record environment response
                next_env_state, reward, terminated, truncated, info  = env.step(action)

                # Get model state based on state variable discretization
                next_model_state = get_discretized_state(next_env_state, discretized_state_vars)

                # Check for termination or episode truncation
                if truncated or terminated:

                    # Modify reward to punish failure
                    reward = 0

                    # Update EnvModel with new transition data and modified reward
                    env_model.update_transitions([(model_state, action, next_model_state, reward)])
                    
                    # End episode
                    break

                # Update EnvModel with new transition data and default reward
                env_model.update_transitions([(model_state, action, next_model_state, reward)])

                # Update env_state
                env_state = next_env_state

                # Display episode summary
                encoded_state = env_model.encoder.encode_state(model_state)
                encoded_action = env_model.encoder.encode_action(action)
                encoded_next_state = env_model.encoder.encode_state(next_model_state)
                transitions_data = env_model.transitions[encoded_state
                                                        ][encoded_action
                                                        ][encoded_next_state]

                logger(tab + f"State:                      {encoded_state}")
                logger(tab + f"Action:                     {encoded_state}")
                logger(tab + f"Next State:                 {encoded_next_state}")
                logger(tab + "Transition Probability:" + f"{transitions_data[0]:.2%}".rjust(8)) 
                logger(tab + f"Transition Expected Reward: {transitions_data[1]:.2f}")
                logger("")

        logger(tab + f"# of unique states visited:     {len(visited_states)}/{n_states}", msg_verbosity=1)
        logger(tab + f"# of new unique states visited: {len(states_visited_this_epoch)}", msg_verbosity=1)






