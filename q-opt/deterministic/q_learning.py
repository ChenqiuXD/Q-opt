import numpy as np
from base_agent import BaseAgent

class q_learning_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R, buffer_size=1000, batch_size=16) -> None:
        super(q_learning_agent, self).__init__(initQ, lr, gamma, P, R, buffer_size, batch_size)

    def update(self):
        """ Update Q-table once """
        # Calculate V at each state
        optimal_actions = np.argmax(self.Q_table, axis=1)
        V_value = np.array( [ self.Q_table[s, optimal_actions[s]] for s in range(self.n_state)] )

        td_error = self.Q_table - self.R - self.gamma * self.P @ V_value
        self.Q_table -= self.lr * td_error
