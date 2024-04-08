# This file is used to randomly generate Markov decision process <S,A,R,P,\gamma>
import numpy as np
from tqdm import trange

class env:
    """ The MDP based environment """
    def __init__(self, n_state, n_action, gamma, low_reward=0, high_reward=10, R_covar=0):
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma

        # Randomly generate transition matrix P and reward matrix R
        self.P = generate_transition(n_state, n_action)
        self.R = generate_reward(n_state, n_action, low=low_reward, high=high_reward)
        self.R_covar = R_covar

        # Set initial state
        self.cur_state = 0

    def reset(self):
        self.cur_state = 0

    def step(self, action):
        """ Randomly sample next state and return immediate reward """
        s = self.cur_state
        reward = self.R[self.cur_state, action] + np.random.normal(loc=0, scale=self.R_covar, size=1)[0]
        next_state_distribution = self.P[self.cur_state][action]
        next_state = np.random.choice(a=self.n_state, p=next_state_distribution)
        self.cur_state = next_state

        return [s, action, reward, next_state]

    def get_state(self):
        return self.cur_state

    def reset(self):
        self.cur_state = 0
        return self.cur_state

    def get_P(self):
        """ Obtain P matrix """
        return self.P
    
    def get_R(self):
        """Obtain R matrix"""
        return self.R
    
    def calc_Q_opt(self):
        """ Return the Q^* function """
        Q_opt = np.zeros([self.n_state, self.n_action])
        Q_ = np.zeros([self.n_state, self.n_action])
        print("Calculating Q-opt in environment")
        for i in range(1000):
            # Calculate V at each state
            optimal_actions = np.argmax(Q_opt, axis=1)
            V_value = np.array( [ Q_opt[s, optimal_actions[s]] for s in range(self.n_state)] )

            # Update Q value of each state-action pair
            for state in range(self.n_state):
                for action in range(self.n_action):
                    Q_[state, action] = self.R[state, action] + self.gamma * self.P[state][action] @ V_value
                    # print("At {}, {}, the value of Q_ is {}, Q_opt is {}".format(state, action, Q_[state, action], Q_opt[state, action]))

            # Break if the differenc is small
            # print("At iteration {} the difference is: {}".format(i, np.max(np.abs(Q_opt-Q_))))
            if np.max(np.abs(Q_opt-Q_))<=1e-2:
                break
            Q_opt = Q_.copy()
        return Q_opt
    
    def evaluate(self, policy):
        """
        This function evaluate the input policy. 
        Input:
            policy: np.ndarray [n_state, n_action], represent the probability of choosing each action at each states. 
        """
        # Compute state transition probabilitym, and reward vector
        n_state = self.n_state
        state_transition_mat = np.zeros([n_state, n_state])
        reward_vec = np.zeros([n_state])
        for s in range(n_state):
            state_transition_mat[s] = policy[s] @ self.P[s]
            reward_vec[s] = policy[s] @ self.R[s]
        
        # Calculate V_\rho (\pi)
        V_vec = np.linalg.inv( np.eye(n_state)-self.gamma*state_transition_mat ) @ reward_vec
        return V_vec[0] # Since env.reset() start from state 0

def generate_transition(n_state, n_action):
    """ Randomly generate transition matrix P (ndarray, [n_state*n_action, n_state]) """
    P = np.zeros([n_state, n_action, n_state])
    for i in range(n_state):
        for j in range(n_action):
            num_positive = np.random.uniform(low=4, high=8, size=1).astype(int)    # Number of transitionable states, other states are set with prob 0
            idx = np.random.choice(n_state, size=num_positive, replace=False)
            prob = np.arange(num_positive) + np.random.uniform(low=-1, high=1, size=num_positive)
            np.random.shuffle(prob)
            P[i][j][idx] = np.exp(prob) / np.sum(np.exp(prob))
            # softmax_val = np.exp(P[i][j]-np.max(P[i][j]))
            # P[i][j] = softmax_val / np.sum(softmax_val)
    return P

def generate_reward(n_state, n_action, low=0, high=10):
    """ Generate reward matrix R (ndarray, [n_state, n_action] ).
      The reward function is designed as: specific state has largest reward.   """
    R = np.ones([n_state, n_action]) * low
    for i in range(n_state):
        idx = np.random.choice(n_action, size=2, replace=False).astype(int)
        R[i, idx[0]] = high
        R[i, idx[1]] = high/2
    return R

if __name__ == "__main__":
    n_action = 3
    n_state = 5
    gamma = 0.8

    environment = env(n_state, n_action, gamma)

    environment.calc_Q_opt()

    for i in range(10):
        s = environment.cur_state
        action = np.random.choice( n_action, p=np.ones(n_action)/n_action )
        r, s_ = environment.step(action)
        print("Env transition from state {} with action {} to state {}, obtaining reward {}".format(s, action, s_, r))
    print("Done")
