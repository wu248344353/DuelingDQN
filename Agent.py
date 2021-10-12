import torch
import random
import numpy as np
import torch.nn as nn
from NetWork import NET
from ReplayBuffer import ReplayBuffer

BATCH_SIZE = 128
GAMMA = 0.99
INITIAL_EPSILON = 1
DECAY_RATE = 1
REPLAY_SIZE = 100000
TARGET_NETWORK_REPLACE_FREQ = 100  # 网络更新频率


class DQN_Agent(object):
    def __init__(self, state_dim, action_dim, isTrain):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE, BATCH_SIZE)
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.IsTrain = isTrain
        # 定义网络，损失函数，优化器
        self.eval_net, self.target_net = NET(self.state_dim, self.action_dim), NET(self.state_dim,  self.action_dim)
        self.eval_net, self.target_net = self.eval_net.to(self.device), self.target_net.to(self.device)
        if self.IsTrain == False:
            self.target_net.load_state_dict(torch.load("net_params.pkl"))  # 加载网络参数
            self.eval_net.load_state_dict(torch.load("net_params.pkl"))  # 加载网络参数
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=5e-5)

    def egreedy_action(self, state):
        state = torch.from_numpy(state).float().view(-1, self.state_dim).to(self.device)
        A_next = self.target_net.advantage(state).detach()
        A_next = A_next.cpu()  # 将数据由gpu转向cpu
        # return np.argmax(Q_next.numpy())
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon - 0.00004
        else:
            self.epsilon *= DECAY_RATE
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            A_next = A_next.cpu()  # 将数据由gpu转向cpu
            return np.argmax(A_next.numpy())

    def action(self, state):
        state = torch.from_numpy(state).float().view(-1, self.state_dim).to(self.device)
        A_next = self.target_net.advantage(state).detach().cpu().numpy()
        return np.argmax(A_next)

    def train(self):
        if self.time_step % TARGET_NETWORK_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.target_net.state_dict(), "net_params.pkl")
        self.time_step += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()
        state_batch = torch.tensor(state_batch).view(BATCH_SIZE, -1).to(self.device)
        action_batch = torch.tensor(action_batch).view(BATCH_SIZE, 1).to(self.device)  # 转换成batch*1的tensor
        reward_batch = torch.tensor(reward_batch).view(BATCH_SIZE, 1).to(self.device)  # 转换成batch*1的tensor
        next_state_batch = torch.tensor(next_state_batch).view(BATCH_SIZE, -1).to(self.device)
        done_batch = torch.tensor(done_batch).view(BATCH_SIZE, 1).to(self.device)

        Q_eval = self.eval_net(state_batch).gather(1, action_batch)  # (batch_size, 1), eval中动作a对应的Q值
        Q_next = self.target_net(next_state_batch).detach()  # 下一个状态的Q值，并且不反向传播
        Q_target = reward_batch + (1 - done_batch) * GAMMA * Q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_fun(Q_eval, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

    def save_paramaters(self, name):
        torch.save(self.target_net.state_dict(), name)
        print("victor:", name)
