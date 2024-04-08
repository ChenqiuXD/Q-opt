import numpy as np

class BaseAgent:
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, buffer_size=1000, batch_size=16) -> None:
        self.n_state = initQ.shape[0]
        self.n_action = initQ.shape[1]
        self.gamma = gamma
        self.Q_table = initQ

        # Step size 
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = np.zeros([buffer_size, 5]) # 5 = dim_state*2 + dim_action + 2 (last two are reward and done)
        self.buffer_cnt = 0

    def choose_action(self, state, epsilon):
        if np.random.uniform() > epsilon:  # 0.1 greedy
            action_value = self.Q_table[state]
            action_list = np.where(action_value == np.max(action_value))[0]
            action = np.random.choice(action_list)
        else:  # random choice
            action = np.random.choice(np.arange(self.n_action))
        return action

    def get_Q(self):
        """ return Q-table """
        return self.Q_table
    
    def store_transition(self, s, a, r, s_, done):
        self.buffer[self.buffer_cnt%self.buffer_size] = np.array([s,a,r,s_,done])
        self.buffer_cnt += 1

    def get_policy(self):
        max_a = np.argmax(self.Q_table, axis=1)
        policy = np.zeros_like(self.Q_table)
        policy[np.arange(self.n_state), max_a] = 1
        return policy