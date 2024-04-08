import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
import gym
import os

# Import algorithm 
# Possible algorithm names "value_iteration", "q_learning", "grad_q", "double_q_learning", "actor_critic"
algo_list = ["q_learning", "double_q_learning"]

algorithms = []
for algo_name in algo_list:
    if algo_name == "q_learning":
        from dqn import DQN as Algo
    elif algo_name == "double_q_learning":
        from ddqn import DDQN as Algo
    elif algo_name == "grad_q":
        from grad_q import grad_q_agent as Algo
    else:
        raise RuntimeError(" The name {} is not included, please recheck name ".format(algo_name))
    algorithms.append(Algo)

# Constant setup

N_EPISODE = 700
AVG_T = 3
reward_sum_traj = np.zeros([len(algorithms), AVG_T, N_EPISODE])
batch_size = 64
buffer_size = 10000
lr = 1e-4
 
# Run simulation
# environment_name could be "LunarLander-v2", "CartPole-v1", "Acrobot-v1", "MountainCar-v0"
environment_name = "Acrobot-v1"
env = gym.make(environment_name, render_mode="rgb_array")
# env = gym.make(environment_name, render_mode="human")

observation = env.reset()

for algo_idx, algo in enumerate(algorithms):
   with tqdm(total=AVG_T*N_EPISODE) as pbar:
      for avg_idx in range(AVG_T):
         # Initialize agent
         agent = algo(n_actions=env.action_space.n,
                     n_features=env.observation_space.shape[0], 
                     learning_rate=lr, 
                     memory_size=buffer_size, 
                     batch_size=batch_size)
         
         # Warmup
         observation = env.reset()
         for t in range(batch_size*5):
            action = env.action_space.sample()
            observation_, reward, terminated, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, terminated)
            observation = observation_

            if terminated:
                observation = env.reset()
         
         # Run an episode
         observation = env.reset()
         for epi_idx in range(N_EPISODE):
            reward_sum = 0
            while True:
                action = agent.choose_action(observation)
                observation_, reward, terminated, info = env.step(action)
                agent.store_transition(observation, action, reward, observation_, terminated)

                agent.learn()

                observation = observation_
                reward_sum += reward

                # env.render()

                if terminated:
                    break
            env.reset()

            reward_sum_traj[algo_idx, avg_idx, epi_idx] = reward_sum
            pbar.update()

env.close()

# Plot using the range of max and minimum
colors = ['b', 'r', 'c', 'm', 'b', 'y']

# Uncertainty line
plt.figure(figsize=[16,6])
for idx, algo in enumerate(algo_list):
    traj = reward_sum_traj[idx]
    # traj -= np.tile(traj[:,0].reshape([-1,1]), N_EPISODE)   # Set all reward sum to initially zero.
    mean_line = np.sum(traj, axis=0) / AVG_T
    plt.plot(np.arange(N_EPISODE), mean_line, label=algo, color=colors[idx])

    low_CI_bound, high_CI_bound = st.t.interval(0.75, N_EPISODE-1, loc=mean_line, scale=st.sem(traj))
    plt.fill_between(np.arange(N_EPISODE), low_CI_bound, high_CI_bound, alpha=0.2, color=colors[idx])
plt.xlabel("Episodes")
plt.ylabel("Reward sum")
plt.title("Reward trajectory with repeated time {}, in environment {}".format(AVG_T, environment_name))
plt.legend()
now = datetime.now()
file_name = str(now.time())+"_figure.png"
file_name = os.path.join(os.path.dirname(__file__), file_name)
plt.savefig(file_name)
print("Image saved in "+file_name)