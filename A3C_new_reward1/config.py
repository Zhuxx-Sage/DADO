import torch
from torch.utils.tensorboard import SummaryWriter

from Utils import *
from environments.TrafficEnvironment import TrafficEnvironment


class Config:
    road_matrix = [[0, Edge(0, 0, 1, 4, 1), Edge(1, 0, 2, 5, 1),
                    Edge(2, 0, 3, 3, 1)],
                   [Edge(3, 1, 0, 8, 1), 0, 0, Edge(4, 1, 3, 5, 1)],
                   [Edge(5, 2, 0, 6, 1), 0, 0, Edge(6, 2, 3, 2, 1)],
                   [Edge(7, 3, 0, 7, 1), Edge(8, 3, 1, 4, 1),
                    Edge(9, 3, 2, 12, 1), 0]]

    urban = UrbanNetGraph(road_matrix, 4, 10)

    ENV = TrafficEnvironment(urban)  # 环境
    observation_input = 14440
    action_output = 10
    GAMMA = 1  # 折扣率
    NUM_EPISODES = 5  # 训练回合数

    NUM_PROCESSES = 1  # agent数
    NUM_ADVANCED_STEP = 5  # 提前计算奖励总和的步数


    value_loss_coef = 1e-7
    entropy_coef = 0.01
    max_grad_norm = 0.5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    # writer = SummaryWriter('loss')
    MAX_FRAMES = int(1e7 / NUM_PROCESSES / NUM_ADVANCED_STEP)
