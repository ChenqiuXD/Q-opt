import numpy as np
from base_agent import BaseAgent

class q_learning_agent(BaseAgent):
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, buffer_size=1000, batch_size=16) -> None:
        super(q_learning_agent, self).__init__(initQ, lr, gamma, buffer_size, batch_size)

    def update(self):
        """ Update Q-table once """
        if self.buffer_cnt > self.batch_size:   # wait for warm up
            # Sample data
            sample_index = np.random.choice( np.minimum(self.buffer.shape[0], self.buffer_cnt), self.batch_size )
            batch_memory = self.buffer[sample_index, :]
            batch_state = batch_memory[:, 0].astype(int)
            batch_action = batch_memory[:, 1].astype(int)
            batch_reward = batch_memory[:, 2]
            batch_next_state = batch_memory[:, 3].astype(int)
            batch_done = (1-batch_memory[:, -1]).astype(int)

            # Compute td target and upate Q table
            q_next = self.Q_table[batch_next_state]
            q_target = batch_reward + self.gamma * batch_done.T * q_next.max(1)
            for idx, state in enumerate(batch_state):
                self.Q_table[state][batch_action[idx]] = (1-self.lr) * self.Q_table[state][batch_action[idx]] + self.lr * q_target[idx]
