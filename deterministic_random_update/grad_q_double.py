import numpy as np
from base_agent import BaseAgent

class double_grad_q_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R) -> None:
        super(double_grad_q_agent, self).__init__(initQ, lr, gamma, P, R)

        self.Q_prime_table = initQ

    def update(self, noise):
        """ Update Q-table once """
        # Calculate V at each state
        optimal_actions = np.argmax(self.Q_prime_table, axis=1)
        V_value = np.array( [ self.Q_prime_table[s, optimal_actions[s]] for s in range(self.n_state)] )

        # Compute TD error and update Q-table by original q learning update rule
        td_error = self.Q_table - self.R - self.gamma * self.P @ V_value + noise
        self.Q_table -= self.lr * td_error
        
        # Update Q_prime table by newly added gradient and a term driving towards original Q-table
        lr = self.lr * 1.0
        for state in range(self.n_state):        
            prev_prob = self.P[:,:,state].reshape([-1])
            grad_new = self.gamma * prev_prob @ td_error.reshape([-1]) # Newly added gradient term for proposed algorithm
            self.Q_table[state, optimal_actions[state]] += lr * self.gamma * grad_new
        self.Q_prime_table += lr * (self.Q_table - self.Q_prime_table)