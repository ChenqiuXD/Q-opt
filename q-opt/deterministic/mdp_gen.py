# This file is used to randomly generate Markov decision process <S,A,R,P,\gamma>
import numpy as np

class env:
    """ The MDP based environment """
    def __init__(self, n_state, n_action, gamma):
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma

        # Randomly generate transition matrix P and reward matrix R
        self.P = generate_transition(n_state, n_action)
        self.R = generate_reward(n_state, n_action)

        # Set initial state
        self.cur_state = 0

    def reset(self):
        self.cur_state = 0

    def step(self, action):
        """ Randomly sample next state and return immediate reward """
        s = self.cur_state
        reward = self.R[self.cur_state, action]
        next_state_distribution = self.P[self.cur_state][action]
        next_state = np.random.choice(a=self.n_state, p=next_state_distribution)
        self.cur_state = next_state

        return np.array([s, action, reward, next_state])

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
    
    def evaluate(self, policy):
        """
        This function evaluate the input policy. 
        Input:
            policy: np.ndarray [n_state, n_action], represent the probability of choosing each action at each states. 
        """
        
    
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

def generate_transition(n_state, n_action):
    """ Randomly generate transition matrix P (ndarray, [n_state*n_action, n_state]) """
    P = np.random.uniform(low=0, high=100, size=[n_state, n_action, n_state])
    for i in range(n_state):
        for j in range(n_action):
            P[i][j] = P[i][j] / np.sum(P[i][j])
    return P

def generate_reward(n_state, n_action):
    """ Randomly generate reward matrix R (ndarray, [n_state, n_action] )  """
    R = np.random.uniform(low=0, high=10, size=[n_state, n_action])
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
