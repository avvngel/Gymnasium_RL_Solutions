#!/usr/bin/env python3
import numpy as np

class GridWorldEnv():

    def __init__(self, n):
        """ Create a nxn GridWorld map. The terminal states (considered as one)
            are the upper left corner and bottom right. The actions are encoded
            as follows:
              - 0: up
              - 1: down
              - 2: left
              - 3: right
        """

        up = 0
        down = 1
        left = 2
        right = 3
        self.n = n
        self.states = list(range(n**2))
        self.grid = self.states[1:n**2 - 1]
        self.terminal_states = [self.states[0], self.states[-1]]
        self.action_space = {up, down, left, right}
        self.state = None
        self.standard_reactions = {up: -n,
                                   down: n,
                                   left: -1,
                                   right: 1}
        self.action_reprs  = {
                              up : '↑',
                              down : '↓',
                              left : '←',
                              right : '→'
                             }

        self.reward = -1
        self.left_col = [i for i in self.states if i%n == 0]
        self.right_col = [i - 1 for i in self.left_col]
        self.top_row = [i for i in range(1, n)]
        self.bottom_row = [i for i in range(n*(n - 1), n**2 - 1)]
        self.terminated = None
        self.transitions = self.initialize_transitions()

    def reset(self):
        """
        Resets environment to a random initial state and reset the terminated 
        flag to False. Must run this function before beginning a simuation.
        """
        self.state = np.random.choice(self.grid)
        self.terminated = False

        return self.state

    def step(self, action):
        """
        Computes environments reaction to a given action and returns
        the next_state, a boolean indicating whether a terminal state
        has been reached, and the reward for the transition
        """
        reward = -1
        reaction = self.standard_reactions[action]
        next_state = self.state + reaction
        if next_state in self.orth_adj_states(self.state):
            self.state = next_state
        if self.state in self.terminal_states:
            reward = 0
            self.terminated = True
        # No implementation
        truncated = None
        info = None

        return self.state, reward, self.terminated, truncated, info
        
    def get_action_space(self):
        """
        Returns a list of actions
        """
        return list(self.action_space)

    def get_available_actions(self, state):
        #available_actions = []
        #for action in self.action_space:
        #    reaction = self.standard_reactions[action]
        #    next_state = state + reaction
        #    if next_state in self.orth_adj_states(state):
        #        available_actions.append(action)
        #return available_actions
        return [0, 1, 2, 3]

    def initialize_transitions(self):

        """
        Returns a list of tuples describing the transitions dynamics 
        of the environment of the form:

        params = [(state, action, next_state,  prob, reward)]
        """

        prob = 1 # Deterministic transition probabilitt
        transitions = [] # Initialize empty transitions list

        # Loop over all possible state->action->next_state transitions
        for state in self.grid:
            for action in list(self.action_space):

                # Get environment reaction for given action
                reaction = self.standard_reactions[action]

                # Compute next_state
                next_state = state + reaction

                # Check for terminal state
                if next_state in self.terminal_states:
                    reward = 0 # Reaching a terminal state is not punished
                else:
                    reward = -1 # Non-terminal transitions are punished

                # Check for case where the agent runs into a wall
                if next_state not in self.orth_adj_states(state):
                    next_state = state # Running into a wall does not change state

                transitions.append([state, action, next_state, prob, reward])
                
                # Print Debugging info
                print("------------------------")
                print(f"State {state}")
                print(f"Action: {self.action_reprs[action]}")
                print(f"Next_State: {next_state}")
                print(f"Probability: {prob}")
                print(f"Reward: {reward}")
        return transitions

    def orth_adj_states(self, state):
        """
        Returns a list of orthogonally adjacent states.
        """
        state_above = state - self.n
        state_below = state + self.n
        state_left = state - 1
        state_right = state + 1
        neighbors = {state_above,
                     state_below,
                     state_left,
                     state_right}
        # Check for edge cases
        if state in self.left_col:
            next_states = neighbors - {state_left}
        elif state in self.right_col:
            next_states = neighbors - {state_right}
        elif state in self.top_row:
            next_states = neighbors - {state_above}
        elif state in self.bottom_row:
            next_states = neighbors - {state_below}
        else:
            # Default case
            next_states = neighbors

        return [i for i in next_states if self.is_on_grid(i)]

    def is_on_grid(self, state):
        return state in set(self.grid) | set(self.terminal_states)

