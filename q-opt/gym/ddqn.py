# Implementation of DQN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class QNetwork(nn.Module):
    def __init__(self, n_features, n_actions, l1=256, l2=128):
        super(QNetwork, self).__init__()
        # Construct network - three layers
        self.fc1 = nn.Linear(n_features, l1)    # input is [n_features*1]
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(l2, n_actions)     # output [n_actions*1]
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class DDQN:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 memory_size=1280, batch_size=32, e_greedy_increment=None,
                 l1=256, l2=128):
        # Parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment

        # Memory
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3)) # 3: a, r, d

        # Construct network
        self.q_network = QNetwork(n_features, n_actions, l1, l2)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.SGD(self.q_network.parameters(),
                                         lr=learning_rate,
                                         momentum=0.9)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, [a, r], s_, [d]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        if np.random.uniform() < self.e_greedy:  # greedy
            action_value = self.q_network(torch.FloatTensor(s)).detach().numpy()
            action_list = np.where(action_value == np.max(action_value))[0]
            action = np.random.choice(action_list)
        else:  # random choice
            action = np.random.choice(np.arange(self.n_actions), 1).item()
        return action

    def learn(self):
        #sample batch from memory
        sample_index = np.random.choice(np.minimum(self.memory_size, self.memory_counter), self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_action = torch.LongTensor(batch_memory[:, self.n_features:self.n_features+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_features+1:self.n_features+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.n_features-1:-1])
        batch_done = torch.LongTensor(batch_memory[:, -1].astype(int))

        #q_eval
        q_eval = self.q_network(batch_state).gather(1, batch_action)
        with torch.no_grad():
            q_next= self.target_q_network(batch_next_state).detach()
            q_target = batch_reward + self.gamma * ((1-batch_done) * q_next.max(1)[0]).view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self, tau=1e-2):
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau*local_param+(1.0-tau) * target_param.data)

    def save_network(self):
        torch.save(self.q_network, 'net.pth')
