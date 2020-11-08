import numpy
import random
import math

from Utils import *
import Config

# SUPER PARAMETERS
A = 0.15
B = 4


class Environment:
    def __init__(self,
                 config: Config
                 ):
        '''
        :param config:  环境配置
        '''

        self.road_network = config.road_network
        self.min_action = config.min_action
        self.max_action = config.max_action
        self.zones_nums = self.road_network.vertex_nums  # 城市区域数量
        self.road_nums = self.road_network.edges_nums  # 城市道路数量
        self.decision_time = config.decision_time
        self.single_time = config.single_time
        self.deadline_nums = self.decision_time  # 为车辆分配的deadline
        self.t = 0  # 时间步上标

        self.state = None  # 状态空间
        self.action = None  # 动作空间

    def reset(self):
        '''
        :return: 重置环境
        '''

    def step(self,
             action):
        '''
        :param action: Agent执行的动作
        :return: next_state，reward，done，info
        '''
        pass
