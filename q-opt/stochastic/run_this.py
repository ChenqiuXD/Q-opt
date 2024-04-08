import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
import os

# from mdp_gen import env
from Maze import Maze as env
from Maze import state2idx

# Construct the environment
np.random.seed(10)
# R_covar = 25
gamma = 0.9
epsilon = 0.2
# environment = env(n_state=n_state, n_action=n_action, gamma=gamma, high_reward=10, low_reward=0, R_covar=R_covar)
environment = env()
n_state = environment.n_states
n_action = environment.n_actions
real_Q = environment.calc_Q_opt(gamma)

# Import algorithm
# Possible algorithm names "value_iteration", "q_learning", "grad_q", "double_q_learning"
# algo_list = ["q_learning", "grad_q", "double_q_learning", "double_grad_q"]
algo_list = ["q_learning", "double_q_learning"]

algorithms = []
for algo_name in algo_list:
    if algo_name == "value_iteration":
        raise RuntimeError("Currently running stochastic version value-based methods, cannot use value iteration")
    elif algo_name == "q_learning":
        from q_learning import q_learning_agent as Algo
    elif algo_name == "grad_q":
        from grad_q import grad_q_agent as Algo
    elif algo_name == "double_q_learning":
        from q_learning_double import double_q_learning_agent as Algo
    elif algo_name == "double_grad_q":
        from grad_q_double import double_grad_q_agent as Algo
    else:
        raise RuntimeError(" The name {} not included, please recheck name ".format(algo_name))
    algorithms.append(Algo)



def train_algorithm():
    # Warmup
    state = state2idx( environment.reset() ) 
    # Main loop:
    state = state2idx( environment.reset() )
    for t in range(T):
        environment.render()
        # Choose action
        if algo.buffer_cnt <= batch_size:
            action = algo.choose_action(state, 1)   # Purely random sample actions
        else:
            action = algo.choose_action(state, epsilon) # eps-greedy policy

        # Store transition
        r, s_, done = environment.step(action)
        s_ = state2idx(s_)
        algo.store_transition(state, action, r, s_, done)
        state = s_

        # Learn
        if algo.buffer_cnt > batch_size:
            algo.update()

            # Record error and Q table
            error = np.mean(np.abs( real_Q - algo.get_Q().flatten() ))
            error_traj[idx, average_idx, t] = error
            Q_traj[idx, average_idx, t] = algo.get_Q()

            # Evaluate the policy
            if t % eval_interval == 0:
                reward_sum_traj[idx, average_idx, int(t/eval_interval)] = environment.evaluate(algo)

            if error <= 1e-2:
                print("Algorithm converged. ")
                break

        # Update process bar
        pbar.update(1)
            
        if done:
            state = state2idx(environment.reset())
    return error_traj, reward_sum_traj, Q_traj

T = 300
AVG_T = 3
eval_interval = 10
error_traj = np.zeros([len(algorithms), AVG_T, T])
reward_sum_traj = np.zeros([len(algorithms), AVG_T, np.ceil(T/eval_interval).astype(int)])
Q_traj = np.zeros([len(algorithms), AVG_T, T, n_state, n_action])
batch_size = 64
buffer_size = int(1e5)
lr = 5e-4

for idx, Algo in enumerate(algorithms):
    with tqdm(total=AVG_T*T) as pbar:
        for average_idx in range(AVG_T):
            initQ = np.random.uniform(low=0, high=20, size=[n_state, n_action])
            algo = Algo(initQ=initQ, lr=lr, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size)

            error_traj, reward_sum_traj, Q_traj = train_algorithm()

colors = ['b', 'r', 'c', 'm', 'b', 'y']
print("Plotting")
plt.figure(figsize=[18,6])

# Plot temporal difference error
plt.subplot(1,2,1)
mean_line = np.sum(error_traj, axis=1) / AVG_T
for idx, algo in enumerate(algo_list):
    plt.plot(np.arange(T), mean_line[idx], label=algo)

    low_CI_bound, high_CI_bound = st.t.interval(0.99, T-1, loc=mean_line[idx], scale=st.sem(error_traj[idx]))
    plt.fill_between(np.arange(T), low_CI_bound, high_CI_bound, alpha=0.2)
plt.xlabel("Iteration")
plt.ylabel("TD error")
plt.legend()

# Plot evaluation result
plt.subplot(1,2,2)
mean_line = np.sum(reward_sum_traj, axis=1) / AVG_T
length = reward_sum_traj.shape[-1]
for idx, algo in enumerate(algo_list):
    plt.plot(np.arange(length), mean_line[idx], label=algo)

    low_CI_bound, high_CI_bound = st.t.interval(0.99, length-1, loc=mean_line[idx], scale=st.sem(reward_sum_traj[idx]))
    plt.fill_between(np.arange(length), low_CI_bound, high_CI_bound, alpha=0.2)
plt.xlabel("Iteration")
plt.ylabel("reward sum")
plt.legend()

# plt.suptitle("n_state={}, n_action={}, R_covar={}, lr={}".format(n_state, n_action, R_covar, lr)) # For mdp_gen.py generated mdp

now = datetime.now()
file_name = now.strftime('%Y-%m-%d-%H:%M:%S')+".png"
file_name = os.path.join(os.path.dirname(__file__), file_name)
plt.savefig(file_name)
# plt.show()
plt.close()