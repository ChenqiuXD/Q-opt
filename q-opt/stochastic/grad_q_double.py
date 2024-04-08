import numpy as np
import copy
from base_agent import BaseAgent

class double_grad_q_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, buffer_size=1000, batch_size =16) -> None:
        super(double_grad_q_agent, self).__init__(initQ, lr, gamma, buffer_size, batch_size)
        # self.Q_prime_table = copy.deepcopy(initQ)
        self.Q_prime_table = np.ones(shape=[self.n_state, self.n_action]) * 100
        self.td_error_table = np.zeros_like(initQ)

    def update(self):
        """ Update Q-table once """
        Q_ = copy.deepcopy(self.Q_table)
        Q_prime_ = copy.deepcopy(self.Q_prime_table)

        if self.buffer_cnt > self.batch_size:   # wait for warm up
            # Sample data
            sample_index = np.random.choice( np.minimum(self.buffer.shape[0], self.buffer_cnt), self.batch_size )
            batch_memory = self.buffer[sample_index, :]
            batch_state = batch_memory[:, 0].astype(int)
            batch_action = batch_memory[:, 1].astype(int)
            batch_reward = batch_memory[:, 2]
            batch_next_state = batch_memory[:, 3].astype(int)

            # Compute TD error and update Q-table by original q learning update rule
            q_next = self.Q_prime_table[batch_next_state]
            q_target = batch_reward + self.gamma * q_next.max(1)
            for idx, s in enumerate(batch_state): 
                a, s_ = batch_action[idx], batch_next_state[idx]
                error = self.Q_table[s,a] - q_target[idx]
                self.td_error_table[s,a] += self.lr * 10 * ( error - self.td_error_table[s,a] )
                Q_[s, a] -= self.lr * error # Q-learning update

                opt_a_s_ = np.argmax(self.Q_prime_table[s_])
                Q_[s_, opt_a_s_] += self.lr * 0.2 * self.gamma * self.td_error_table[s, a]   # Newly added grad
                # Q_[s_, opt_a_s_] += self.lr * 1. * self.gamma * error   # Newly added grad

                # soft update Q_prime towards Q
                tau = 1e-3
                Q_prime_[s,a] += tau * (self.Q_table[s,a] - self.Q_prime_table[s,a])
            self.Q_table = Q_.copy()
            self.Q_prime_table = Q_prime_.copy()