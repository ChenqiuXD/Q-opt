import numpy as np

class value_iteration_agent:
    """ Use value iteration to find optimal policy """
    def __init__(self, initQ, lr, gamma, P, R) -> None:
        self.n_state = initQ.shape[0]
        self.n_action = initQ.shape[1]
        self.gamma = gamma
        self.Q_table = initQ

        # Environment related parameters
        self.P = P  # transition matrix
        self.R = R  # Reward matrix

    def update(self):
        """ Update Q-table once """
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