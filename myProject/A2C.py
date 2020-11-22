import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical, normal, MultivariateNormal
import numpy as np
from config import Config

config = Config()


def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Net(nn.Module):

    def __init__(self, n_in, n_out):
        super(Net, self).__init__()

        def init_(module): return init(
            module, gain=nn.init.calculate_gain('relu'))

        def init_(module): return init(module, gain=1.0)

        self.s_dim = n_in
        self.a_dim = n_out
        self.a1 = init_(nn.Linear(n_in, 1200))
        self.mid_a = init_(nn.Linear(1200, 360))
        self.mu = init_(nn.Linear(360, n_out))
        self.sigma = init_(nn.Linear(360, n_out))

        self.c1 = init_(nn.Linear(n_in, 360))
        self.mid_c = init_(nn.Linear(360, 100))
        self.v = init_(nn.Linear(100, 1))

        self.distribution = torch.distributions.Normal

        self.train()
        # self.fc1 = nn.Linear(n_in, n_mid)
        # self.fc2 = nn.Linear(n_mid, n_out)
        # self.actor = nn.Linear(n_mid, n_out)  # 动作输出，动作的数量
        #
        # self.critic = nn.Linear(n_mid, 1)  # 状态价值，输出1

    def forward(self, x):  # size[14440]
        x = x.view(-1, 1444)  # [10,1444]
        a1 = torch.tanh(self.a1(x))
        mid_a = torch.tanh(self.mid_a(a1))
        # mu = 2 * torch.tanh(self.mu(a1))
        mu = 2 * torch.tanh(self.mu(mid_a))
        sigma = F.softplus(self.sigma(mid_a)) + 0.001
        c1 = F.relu6(self.c1(x))
        mid_c = F.relu6(self.mid_c(c1))
        values = self.v(mid_c)

        return mu, sigma, values
        # x = x.reshape(x.shape[0], -1)
        # x = x.float()
        # h1 = F.tanh(self.fc1(x))
        # h2 = F.tanh(self.fc2(h1))
        # critic_output = self.critic(h2)  # 状态价值计算
        # actor_output = self.actor(h2)  # 动作计算
        #
        # return critic_output, actor_output

    def act(self, x):
        self.training = False
        mu, sigma, _ = self.forward(x)
        # m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        m = self.distribution(mu.data, sigma.data)
        return m.sample()
        # '''按概率求状态x的动作'''
        # value, actor_output = self(x)
        # means = actor_output[:, :1].squeeze(0)
        # stds = actor_output[:, 1:].squeeze(0)
        # if len(means.shape) == 2:
        #     means = means.squeeze(-1)
        # if len(stds.shape) == 2:
        #     stds = stds.squeeze(-1)
        # if len(stds.shape) > 1 or len(means.shape) > 1:
        #     raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        # action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
        # action = action_distribution.sample().cpu().numpy()
        # action = np.clip(action, 0, 6)
        # return action, action_distribution

    def get_value(self, x):
        '''获得状态价值'''
        mu, sigma, value = self.forward(x)
        return value  # torch.size[10,1]

    def evaluate_actions(self, x, actions):
        '''从状态x获取状态值，记录实际动作的对数概率和熵'''
        mu, sigma, value = self.forward(x)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(actions)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)

        return value, log_prob, entropy
