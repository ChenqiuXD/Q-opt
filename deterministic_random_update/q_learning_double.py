import numpy as np
from base_agent import BaseAgent

class double_q_learning_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R, buffer_size=1000, batch_size=16) -> None:
        super(double_q_learning_agent, self).__init__(initQ, lr, gamma, P, R, buffer_size, batch_size)

        self.Q_prime_table = initQ

    def update(self, noise):
        """ Update Q-table once """
        # Calculate V at each state
        optimal_actions = np.argmax(self.Q_prime_table, axis=1)
        V_value = np.array( [ self.Q_prime_table[s, optimal_actions[s]] for s in range(self.n_state)] )

        # Update Q-value
        td_error = self.Q_table - self.R - self.gamma * self.P @ V_value
        Q_ = self.Q_table - self.lr * (td_error + noise)

        # Use random coordinate update
        # rand_state = np.random.choice(np.arange(self.n_state), size=int(self.n_state/3), replace=False)
        # self.Q_table[rand_state] = Q_[rand_state]
        rand_action = np.random.choice(np.arange(self.n_action), size=int(self.n_action/3), replace=False)
        self.Q_table[:, rand_action] = Q_[:, rand_action]
                
        # Update target Q-value
        self.Q_prime_table += self.lr * (self.Q_table - self.Q_prime_table)
