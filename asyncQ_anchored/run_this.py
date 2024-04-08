"""
Test environment with seven grid world. Agent are born at state 2, and the process terminate when choosing to move right at state 6.
The reward function is very simple, each step will lead to reward -1.
--------------------------------------
| 0 | 1 | 2 | 3 | 4 | terminal|
--------------------------------------
The inital Q is randomly generated and we compare the result of Q-learning and double Q-learning
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

colors = ['b', 'c', 'g', 'k', 'm', 'r', 'y', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
num_state = 5

def operatorT(Q, gamma):
    """ The corresponding operator T with grid world setting """
    assert np.all(Q.shape==np.array([num_state, 2]))    # [n_state, n_action]
    TQ = copy.deepcopy(Q)
    for s in np.arange(num_state-2)+1:
        TQ[s,0] = -1 + gamma * np.max(Q[s-1])    # Move left
        TQ[s,1] = -1 + gamma * np.max(Q[s+1])    # Move right
    TQ[0,0] = -1 + gamma * np.max(Q[0])  # Move left at state 0 will return to state 0
    TQ[0,1] = -1 + gamma * np.max(Q[1])
    TQ[num_state-1,0] = -1 + gamma * np.max(Q[num_state-2])  
    TQ[num_state-1,1] = -1     # Move right at state 6 terminate this process

    return TQ

def main():
    """
    Value iteration method, directly apply T operator repeatedly to the vector Q
    """
    # # Hyper paramters
    # T = 10
    # gamma = 0.9

    # # Update Q
    # Q = np.random.uniform(low=-10, high=0, size=[num_state,2])
    # Q_history_value_iteration = np.zeros([T, Q.shape[0], Q.shape[1]])
    # for t in range(T):
    #     Q_history_value_iteration[t] = Q
    #     Q_next = operatorT(Q, gamma)
    #     Q = Q_next

    # # Plot lines
    # plt.figure()
    # for s in np.arange(num_state):
    #     for a in [0,1]:
    #         plt.plot(np.arange(T), Q_history_value_iteration[:, s, a], label="s{}-a{}".format(s, a))
    # plt.legend(loc="upper right")
    # plt.show()






    # """
    # Q-learning and double Q learning method without stochasticity
    # """
    # Hyper paramters
    # T = 2000
    # gamma = 0.9
    # lr = 1e-2
    # lr_soft_update = 0.1

    # # Initialization
    # Q_init = np.random.uniform(low=-10, high=0, size=[num_state,2])
    # Q_history_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])
    # Q_history_double_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])

    # # Q-learning
    # Q = copy.deepcopy(Q_init)
    # Q_d = copy.deepcopy(Q_init)
    # Q_target = copy.deepcopy(Q_init)
    # for t in range(T):
    #     Q_history_Q_learning[t] = Q
    #     Q_history_double_Q_learning[t] = Q_d

    #     Q_next = operatorT(Q, gamma)
    #     Q_next_d = operatorT(Q_target, gamma)

    #     Q += lr * (Q_next - Q)
    #     Q_d += lr * (Q_next_d - Q_d)  
    #     Q_target += lr_soft_update * (Q_d - Q_target) 

    # # Plot lines
    # plt.figure(figsize=[14, 6])

    # cnt = 0
    # for s in np.arange(num_state):
    #     for a in [0,1]:
    #         plt.plot(np.arange(T), Q_history_Q_learning[:, s, a], color=colors[cnt], alpha=0.7, linestyle='-', label="s{}-a{}".format(s, a))
    #         plt.plot(np.arange(T), Q_history_double_Q_learning[:, s, a], color=colors[cnt], linestyle='-.')
    #         cnt+= 1
    # plt.title("Q-traj with Q-learning (linestyle '-') and double Q-learning (linestyle '-.')")
    # plt.legend(loc="upper right")

    # plt.show()


    """
    Q-learning and double Q learning method with noise
    """
    # Hyper paramters
    T = 5000
    gamma = 0.9
    lr = 5e-3
    lr_soft_update = 5e-3

    # Initialization
    Q_init = np.random.uniform(low=-10, high=0, size=[num_state,2])
    Q_history_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])
    Q_history_double_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])

    # Q-learning
    Q = copy.deepcopy(Q_init)
    Q_d = copy.deepcopy(Q_init)
    Q_target = copy.deepcopy(Q_init)
    for t in range(T):
        noise = np.random.normal(loc=0, scale=2, size=Q.shape)
        Q_history_Q_learning[t] = Q
        Q_history_double_Q_learning[t] = Q_d

        Q_next = operatorT(Q, gamma) + noise
        Q_next_d = operatorT(Q_target, gamma) + noise

        Q += lr * (Q_next - Q)
        Q_d += lr * (Q_next_d - Q_d)  
        Q_target += lr_soft_update * (Q_d - Q_target) 

    # Plot lines
    fig = plt.figure(figsize=[16,10])
    ax = fig.subplots()
    cnt = 0
    lines = []
    for s in np.arange(num_state):
        line_state = []
        for a in [0,1]:
            p1 = ax.plot(np.arange(T), Q_history_Q_learning[:, s, a], color=colors[cnt], alpha=0.5, linestyle='-', label="s{}-a{}".format(s, a))
            p2 = ax.plot(np.arange(T), Q_history_double_Q_learning[:, s, a], color=colors[cnt], linestyle=':')
            cnt+= 1
            line_state.append([p1, p2])
        lines.append(line_state)
    plt.title("Q-traj with Q-learning (linestyle '-') and double Q-learning (linestyle '.')")
    plt.legend(loc="upper right")

    labels = ["s0", "s1", "s2", "s3", "s4"]
    
    def func(label):
        index = labels.index(label)
        # print(lines[index][1])
        lines[index][0][0][0].set_visible(not lines[index][0][0][0].get_visible())
        lines[index][0][1][0].set_visible(not lines[index][0][1][0].get_visible())
        lines[index][1][0][0].set_visible(not lines[index][1][0][0].get_visible())
        lines[index][1][1][0].set_visible(not lines[index][1][1][0].get_visible())
        fig.canvas.draw()

    # Matplotlib check buttons
    label = [True, True, True, True, True]
    ax_check = plt.axes([0.9, 0.001, 0.2, 0.3])
    plot_button = CheckButtons(ax_check, labels, label)
    plot_button.on_clicked(func)
    
    plt.show()