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
from run_this import colors, num_state, operatorT

# Hyper paramters
T = 10000
gamma = 0.9
lr = 5e-3
lr_soft_update = 5e-3

# Initialization
Q_init = np.random.uniform(low=-10, high=0, size=[num_state,2])
Q_history_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])
Q_history_double_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])
Q_history_anchored_Q_learning = np.zeros([T, Q_init.shape[0], Q_init.shape[1]])

# Q-learning
Q = copy.deepcopy(Q_init)
Q_d = copy.deepcopy(Q_init)
Q_target = copy.deepcopy(Q_init)
Q_anchored = copy.deepcopy(Q_init)
Q_anchored_target = copy.deepcopy(Q_init)
for t in range(T):
    noise = np.random.normal(loc=0, scale=2, size=Q.shape)
    Q_history_Q_learning[t] = Q
    Q_history_double_Q_learning[t] = Q_d
    Q_history_anchored_Q_learning[t] = Q_anchored

    Q_next = operatorT(Q, gamma) + noise
    Q_next_d = operatorT(Q_target, gamma) + noise
    Q_next_anchored = operatorT(Q_anchored, gamma) + noise

    rand_states = np.random.choice(num_state, size=3, replace=True)
    rand_actions = np.random.choice(2, size=3, replace=True)
    
    for i in range(3):
        s = rand_states[i]
        a = rand_actions[i]

        Q[s,a] += lr * (Q_next[s,a] - Q[s,a])

        Q_d[s,a] += lr * (Q_next_d[s,a] - Q_d[s,a])
        Q_target[s,a] += lr_soft_update * (Q_d[s,a] - Q_target[s,a]) 

        Q_anchored[s,a] += lr * (Q_next[s,a]-Q_anchored[s,a]) + lr_soft_update * (Q_anchored_target[s,a] - Q_anchored[s,a]) 
        Q_anchored_target[s,a] += lr_soft_update * (Q_anchored[s,a] - Q_anchored_target[s,a])

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
        p3 = ax.plot(np.arange(T), Q_history_anchored_Q_learning[:, s, a], color=colors[cnt], linestyle='-.')
        cnt+= 1
        line_state.append([p1, p2, p3])
    lines.append(line_state)
plt.title("Q-traj with Q-learning (linestyle '-'), double Q-learning (linestyle ':'), and anchored Q-learning (linestyle '-.')")
plt.legend(loc="upper right")

labels = ["s0", "s1", "s2", "s3", "s4"]

# Plot truth
alpha = 0.5
p1 = ax.plot(np.arange(T), [-(1-gamma**6)/(1-gamma)]*T, alpha=alpha, color=colors[0])  # (s0-a0)
p2 = ax.plot(np.arange(T), [-(1-gamma**5)/(1-gamma)]*T, alpha=alpha, color=colors[1])  # (s0-a1)
lines[0].append([p1, p2])
p1 = ax.plot(np.arange(T), [-(1-gamma**6)/(1-gamma)]*T, alpha=alpha, color=colors[2])  # (s1-a0)
p2 = ax.plot(np.arange(T), [-(1-gamma**4)/(1-gamma)]*T, alpha=alpha, color=colors[3])  # (s1-a1)
lines[1].append([p1, p2])
p1 = ax.plot(np.arange(T), [-(1-gamma**5)/(1-gamma)]*T, alpha=alpha, color=colors[4])  # (s2-a0)
p2 = ax.plot(np.arange(T), [-(1-gamma**3)/(1-gamma)]*T, alpha=alpha, color=colors[5])  # (s2-a1)
lines[2].append([p1, p2])
p1 = ax.plot(np.arange(T), [-(1-gamma**4)/(1-gamma)]*T, alpha=alpha, color=colors[6])  # (s3-a0)
p2 = ax.plot(np.arange(T), [-1-gamma]*T, alpha=alpha, color=colors[7])  # (s3-a1)
lines[3].append([p1, p2])
p1 = ax.plot(np.arange(T), [-(1-gamma**3)/(1-gamma)]*T, alpha=alpha, color=colors[8])  # (s4-a0)
p2 = ax.plot(np.arange(T), [-1]*T, alpha=alpha, color=colors[9])  # (s4-a1)
lines[4].append([p1, p2])
 
def func(label):
    index = labels.index(label)
    # print(lines[index][0])
    lines[index][0][0][0].set_visible(not lines[index][0][0][0].get_visible())
    lines[index][0][1][0].set_visible(not lines[index][0][1][0].get_visible())
    lines[index][0][2][0].set_visible(not lines[index][0][2][0].get_visible())
    lines[index][1][0][0].set_visible(not lines[index][1][0][0].get_visible())
    lines[index][1][1][0].set_visible(not lines[index][1][1][0].get_visible())
    lines[index][1][2][0].set_visible(not lines[index][1][2][0].get_visible())
    fig.canvas.draw()

# Matplotlib check buttons
label = [True, True, True, True, True]
ax_check = plt.axes([0.9, 0.001, 0.2, 0.3])
plot_button = CheckButtons(ax_check, labels, label)
plot_button.on_clicked(func)
plt.show()
