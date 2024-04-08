import numpy as np
import random

class BaseAgent:
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R, buffer_size=1000, batch_size=16) -> None:
        self.n_state = initQ.shape[0]
        self.n_action = initQ.shape[1]
        self.gamma = gamma
        self.Q_table = initQ

        # Environment related parameters
        self.P = P  # transition matrix
        self.R = R  # Reward matrix

        # Step size 
        self.lr = lr
        self.batch_size = batch_size
        self.buffer = np.zeros([buffer_size, 4]) # 5 = dim_state*2 + dim_action + 1 (last two are reward)
        self.learn_cnt = 0

    def update(self):
        raise NotImplementedError

    def get_Q(self):
        """ return Q-table """
        return self.Q_table
    
    def get_policy(self):
        max_a = np.argmax(self.Q_table, axis=1)
        policy = np.zeros_like(self.Q_table)
        policy[:, max_a] = 1
        return policy