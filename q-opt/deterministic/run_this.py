import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as st

from mdp_gen import env

# Construct the environment
np.random.seed(10)
n_state = 10
n_action = 20
gamma = 0.9
environment = env(n_state=n_state, n_action=n_action, gamma=gamma)
real_Q = environment.calc_Q_opt()

# Import algorithm
# Possible algorithm names "value_iteration", "q_learning", "grad_q", "double_q_learning"
# algo_list = ["value_iteration", "q_learning", "double_q_learning", "grad_q", "double_grad_q"]
algo_list = ["q_learning", "value_iteration", "double_q_learning", "grad_q", "double_grad_q"]


algorithms = []
for algo_name in algo_list:
    if algo_name == "value_iteration":
        from value_iteration import value_iteration_agent as Algo
    elif algo_name == "q_learning":
        from q_learning import q_learning_agent as Algo
    elif algo_name == "grad_q":
        from grad_q import grad_q_agent as Algo
    elif algo_name == "double_q_learning":
        from q_learning_double import double_q_learning_agent as Algo
    elif algo_name == "double_grad_q":
        from grad_q_double import double_grad_q_agent as Algo
    elif algo_name == "async_value_iteration":
        from async_value_iteration import async_value_iteration_agent as Algo
    else:
        raise RuntimeError(" The name {} is not included, please recheck name ".format(algo_name))
    algorithms.append(Algo)

"""
Some summary on the experiment: 
1. all the algorithm will converge, to see all the convergece, set the parameters T to 100000
2. The q-learning and double-q learning converge quickliest however q-grad converge faster in the beginning.
"""

T = 1500
AVG_T = 10  # Randomly obtain initQ
eval_interval = 20
error_traj = np.zeros([len(algorithms), AVG_T, T])
reward_sum_traj = np.zeros([len(algorithms), AVG_T, np.ceil(T/eval_interval).astype(int)])
lr = 5e-2
for idx, Algo in enumerate(algorithms):
    with tqdm(total=AVG_T*T) as pbar:
        for average_idx in range(AVG_T):
            initQ = np.random.uniform(low=0, high=200, size=[n_state, n_action])
            algo = Algo(initQ=initQ, lr=lr, gamma=gamma, P=environment.get_P(), R=environment.get_R())

            # Main loop:
            for t in range(T):
                algo.update()
                error = np.max(np.abs( real_Q-algo.get_Q() ))
                error_traj[idx, average_idx, t] = error
                # print("At iteration {}, the error between real Q and algo's Q talbe is {}".format(t, error))

                if t % eval_interval == 0:
                    pi = algo.get_policy()
                    reward_sum_traj = env.evaluate(pi)

                # Update process bar
                pbar.update(1)

                if error <= 1e-2:
                    print("Algorithm converged. ")
                    break

colors = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
plt.figure(figsize=[16,6])

# Plot
mean_line = np.sum(error_traj, axis=1) / AVG_T
for idx, algo in enumerate(algo_list):
    plt.plot(np.arange(T), mean_line[idx], label=algo)

    low_CI_bound, high_CI_bound = st.t.interval(0.99, T-1, loc=mean_line[idx], scale=st.sem(error_traj[idx]))
    plt.fill_between(np.arange(T), low_CI_bound, high_CI_bound, alpha=0.2)

plt.xlabel("Iteration")
plt.ylabel("TD error")
plt.legend()
plt.show()