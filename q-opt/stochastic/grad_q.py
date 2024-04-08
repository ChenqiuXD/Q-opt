import numpy as np
import copy
from base_agent import BaseAgent
from numba import jit

class grad_q_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, buffer_size=1000, batch_size=16) -> None:
        super(grad_q_agent, self).__init__(initQ, lr, gamma, buffer_size, batch_size)
        self.td_error_table = np.zeros_like(initQ)  # Stores temporla differnece error for update of newly added grad

    @jit(nopython=True)
    def update(self):
        """ Update Q-table once """
        Q_ = copy.deepcopy(self.Q_table)

        if self.buffer_cnt > self.batch_size:   # wait for warm up
            # Sample data
            sample_index = np.random.choice( np.minimum(self.buffer.shape[0], self.buffer_cnt), self.batch_size )
            batch_memory = self.buffer[sample_index, :]
            batch_state = batch_memory[:, 0].astype(int) 
            batch_action = batch_memory[:, 1].astype(int)
            batch_reward = batch_memory[:, 2]
            batch_next_state = batch_memory[:, 3].astype(int)

            # Compute TD error and update Q-table by original q learning update rule
            q_next = self.Q_table[batch_next_state]
            q_target = batch_reward + self.gamma * q_next.max(1)
            for idx, s in enumerate(batch_state):
                a, s_ = batch_action[idx], batch_next_state[idx]
                error = self.Q_table[s,a] - q_target[idx]
                self.td_error_table[s,a] += self.lr * 3 * ( error - self.td_error_table[s,a] )
                Q_[s, a] -= self.lr * error # Q-learning update

                opt_a_s_ = np.argmax(self.Q_table[s_])
                Q_[s_, opt_a_s_] += self.lr * self.gamma * self.td_error_table[s, a]   # Newly added grad
                # Q_[s_, opt_a_s_] += self.lr * 1. * self.gamma * error   # Newly added grad
            self.Q_table = Q_.copy()