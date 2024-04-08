import numpy as np
import copy
from base_agent import BaseAgent

class grad_q_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R) -> None:
        super(grad_q_agent, self).__init__(initQ, lr, gamma, P, R)

    def update(self, noise):
        """ Update Q-table once """
        Q_ = copy.deepcopy(self.Q_table)

        # Calculate V at each state
        optimal_actions = np.argmax(self.Q_table, axis=1)
        V_value = np.array( [ self.Q_table[s, optimal_actions[s]] for s in range(self.n_state)] )

        # Update Q-value
        td_error = self.Q_table - self.R - self.gamma * self.P @ V_value + noise
        Q_ -= self.lr * td_error

        # Update Q-table by newly added gradient
        lr = self.lr * 0.4
        for state in range(self.n_state):
            prev_prob = self.P[:,:,state].reshape([-1])
            grad_new = self.gamma * prev_prob @ td_error.reshape([-1]) # Newly added gradient term for proposed algorithm
            Q_[state, optimal_actions[state]] += lr * grad_new
        self.Q_table = Q_.copy()