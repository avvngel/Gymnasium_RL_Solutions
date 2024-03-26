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
        self.grid = [i for i in range(1, n**2 - 1)]
        self.action_space = {up, down, left, right}
        self.state = None
        self.standard_reactions = {up: -n,
                                   down: n,
                                   left: -1,
                                   right: 1}
        self.terminal_states = [0, n**2 - 1]
        self.reward = -1
        self.terminated = None
        self.transitions = self.initialize_transitions()
        self.states = self.grid + self.terminal_states

    def reset(self):
        """
        Resets environment to a random initial state and reset the terminated 
        flag to False. Must run this function before beginning a simuation.
        """
        self.state = np.random.choice(self.grid)
        self.terminated = False

    def step(self, action):
        """
        Computes environments reaction to a given action and returns
        the next_state, a boolean indicating whether a terminal state
        has been reached, and the reward for the transition
        """
        reward = -1
        reaction = self.standard_reactions[action]
        next_state = self.state + reaction
        if self.is_on_grid(next_state):
            self.state = next_state
        if self.state in self.terminal_states:
            reward = 0
            self.terminated = True
        return self.state, self.terminated, reward
        
    def get_action_space(self):
        """
        Returns a list of actions
        """
        return list(self.action_space)

    def get_available_actions(self, state):
        available_actions = []
        for action in self.action_space:
            reaction = self.standard_reactions[action]
            next_state = state + reaction
            if self.is_on_grid(next_state):
                available_actions.append(action)
        return available_actions

    def initialize_transitions(self):
        prob = 1
        transitions = []
        for state in self.grid:
            available_actions = self.get_available_actions(state)
            for action in available_actions:
                reaction = self.standard_reactions[action]
                next_state = state + reaction
                if next_state in self.terminal_states:
                    reward = 0
                else:
                    reward = -1
                transitions.append([state, action, next_state, prob, reward])
        return transitions

    def get_orth_adj_nodes(self, state):
        """
        Returns a list of orthogonally adjacent states.
        """
        orth_adj_nodes = []
        # Check if node to the right is on the grid
        if self.is_on_grid(state + 1):
            adj_nodes.append(state + 1)
        # Check if node to the left is on the grid
        if self.is_on_grid(state - 1):
            adj_nodes.append(state - 1)
        # Check if node above is on the grid
        if self.is_on_grid(state + self.n):
            adj_nodes.append(state + self.n)
        # Check if node below is on the grid
        if self.is_on_grid(state - self.n):
            adj_nodes.append(state - self.n)

        return orth_adj_nodes


    def is_on_grid(self, state):
        return state in set(self.grid) | set(self.terminal_states)


