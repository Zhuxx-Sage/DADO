# 配置文件

from Utils import *
import TrafficEnvironment


class Config:
    def __init__(self):
        self.road_network = UrbanNetGraph()  # 路网
        self.environment = TrafficEnvironment()  # 交通环境
        self.min_action = 0  # 最小收费值
        self.max_action = 6  # 最大收费值
        self.decision_time = 360  # 决策时长，暂定6h=360min，从早上9点-下午3点
        self.single_time = 10  # 单个时间步长，即τ=10min
        self.w = 0.5  # 时间敏感度
        self.w_ = 0.5  # 对成本的敏感度
        self.A = 0.15  # BPR模型参数
        self.B = 4  # BPR模型参数
