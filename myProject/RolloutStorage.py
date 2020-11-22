import torch

from environments.TrafficEnvironment import TrafficEnvironment
from Utils import *

from config import Config
config = Config()



# 存储类
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape).to(config.device)
        self.masks = torch.ones(num_steps +  1, num_processes, 1).to(config.device)
        self.rewards = torch.ones(num_steps, num_processes, 1).to(config.device)
        self.actions = torch.zeros(num_steps, num_processes, 10).long().to(config.device)

        # 存储折扣奖励总和
        self.returns = torch.zeros(num_steps + 1, num_processes, 10).to(config.device)
        self.index = 0 # insert索引

    def insert(self, current_obs, action, reward, mask):
        '''
        :param current_obs:
        :param action:
        :param reward:
        :param mask:
        :return: 存储transition到下一个index
        '''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action.view(1, -1))

        self.index = (self.index + 1) % config.NUM_ADVANCED_STEP  # 更新索引

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''
        :param next_value:
        :return: 计算Advantage的步骤中每个步骤的折扣奖励总和
        '''
        # print(self.returns[-1].shape)
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * config.GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]
