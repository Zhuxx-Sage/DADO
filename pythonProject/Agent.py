import numpy
import torch
import time
import logging

from torch.optim import optimizer
import Config

class Base_Agent(object):

    def __init__(self, config: Config):
        '''
        :param config: 配置文件
        '''
        self.config = config
        self.environment = config.environment
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"

    def reset_game(self):
        self.state = self.environment.rese

    def step(self):
        """Runs a step within a game including a learning step if required"""
        pass

    def pick_and_conduct_action_and_save_log_probablitied(self):

    def pick_action_and_get_log_probablities(self):
        pass

    def
