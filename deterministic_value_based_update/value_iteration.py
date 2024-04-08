import numpy as np
from base_agent import BaseAgent

class value_iteration_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R) -> None:
        super(value_iteration_agent, self).__init__(initQ, lr, gamma, P, R)

    def update(self, noise=None):
        """ Update Q-table once """
        if noise: raise KeyError("noise must be zero in value iteration method. ")
        Q_ = self.Q_table

        # Calculate V at each state
        optimal_actions = np.argmax(self.Q_table, axis=1)
        V_value = np.array( [ self.Q_table[s, optimal_actions[s]] for s in range(self.n_state)] )

        # Update Q value of each state-action pair
        for state in range(self.n_state):
            for action in range(self.n_action):
                Q_[state, action] = self.R[state, action] + self.gamma * self.P[state][action] @ V_value
                
        self.Q_table = Q_.copy()

    def get_Q(self):
        """ return Q-table """
        return self.Q_table