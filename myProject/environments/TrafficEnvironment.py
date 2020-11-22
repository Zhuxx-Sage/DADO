import numpy as np
import random
import math

from numpy import NaN

from Utils import *
from environments.Env import *
import seeding
import numpy as np
import torch


# SUPER PARAMETERS


class TrafficEnvironment(Env):
    observation_space = 4 * 361
    action_space = 1

    def __init__(self, urban: UrbanNetGraph
                 ):
        '''
        :param config:  环境配置
        '''
        self.urban = urban.init_infos()
        self.min_action = 0
        self.max_action = 6
        self.zones_nums = self.urban.vertex_nums  # 城市区域数量
        self.road_nums = self.urban.edges_nums  # 城市道路数量
        self.decision_time = 360  # decision time, 360min
        self.single_time = 10  # time step, 10min
        self.step_nums = int(self.decision_time / self.single_time)
        self.deadline_time = 1  # 为车辆分配的deadline区间， 1min
        self.deadline_nums = int(self.decision_time / self.deadline_time) + 1  # 为车辆分配的deadline

        self.t = 1  # 时间步上标

        self.A = 0.15
        self.B = 4
        self.w = 0.5
        self.w_ = 0.5
        self.D = 10  # 时间阈值，单位min

        self.state = None  # 状态空间
        self.action = None  # 动作空间

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        '''
        :return: 重置环境
        '''
        self.t = 1
        self.state = np.zeros((self.road_nums, self.zones_nums, self.deadline_nums))
        for e_key, e_value in self.urban.Edges.items():
            # road_capacity = 0  # 路的容量
            # 生成zones个数的随机数
            # random_zones_vehicles = np.array(
            #     [self.np_random.randint(int(0.5 * e_value.edge_capacity / self.zones_nums),
            #                             int(0.7 * e_value.edge_capacity / self.zones_nums),
            #                             (self.zones_nums))])
            # # 对每个区域生成d
            # for i in range(self.zones_nums):
            #     e_capacity = random_zones_vehicles[0][i]
            #     while (e_capacity > 0):
            #         random_d_index = self.np_random.randint(0, self.deadline_nums, 1)
            #         random_vehicles = self.np_random.randint(0, e_capacity + 1, 1)
            #         e_capacity -= random_vehicles
            #         self.state[e_value.edge_id, i, random_d_index] = random_vehicles

            # self.state[e_value.edge_id] = np.array(
            #     [self.np_random.randint(int(0.5 * e_value.edge_capacity / self.zones_nums),
            #                             int(0.7 * e_value.edge_capacity / self.zones_nums),
            #                             (self.zones_nums, self.deadline_nums))])
            self.state[e_value.edge_id] = np.array(
                [self.np_random.randint(0, 2,
                                        (self.zones_nums, self.deadline_nums))])
        return np.array(self.state, dtype=int)

    def travel_time_on_road(self,
                            road_e: Edge, ):
        '''
        :param road:  road e
        :return: 返回在道路 e上面行驶的时间
        '''
        now_state = self.state.copy()
        now_state_by_road = now_state[road_e.edge_id]
        sum_by_road = now_state_by_road.sum(axis=(0, 1))  # 按road e进行车流求和，即road e上总的车辆数
        travel_time = road_e.get_edge_free_flow_time() * (
                1 + self.A * (sum_by_road / road_e.get_edge_capacity()) ** self.B)
        return travel_time

    def travel_cost_by_one_path(self,
                                one_path,  # Edge类数组
                                d):  # d:deadline
        '''
        :param ont_path: [Edge(0,1,2,3,4),Edge(2,3,4,5,1)
        :return: 计算一条 道路组成的路径 的行驶成本
        '''
        now_action = np.copy(self.action)
        cost_by_path = 0
        for e in one_path:
            if d == 0:
                cost_by_path += (now_action[e.edge_id] + self.w * self.travel_time_on_road(e))
            else:
                X = d - (self.travel_time_on_road(e) + (self.t - 1) * self.single_time)
                inx = self.D - X
                if X > self.D:
                    cost_by_path += (now_action[e.edge_id] + np.exp(inx))
                else:
                    cost_by_path += np.power(inx, 2)

        return cost_by_path

    def path_preference(self,
                        vi,
                        vj,
                        one_path,
                        d):
        '''
        :param vi: 出发地
        :param vj: 目的地
        :param one_path: 从目的地到出发地的一条路径
        :param d: 车流的截止时间
        :return: 计算一条路径的偏好度
        '''

        # 得到此条路径的cost
        one_path_cost = self.travel_cost_by_one_path(one_path, d)
        exp_one_path_cost = np.exp(-self.w_ * one_path_cost)

        # 得到所有从出发地到目的地的路径
        if vi == vj:
            return 0
        else:
            all_paths = self.urban.get_path_by_complete_edge(vi, vj, p=[])
            exp_all_paths_cost = 0
            for path in all_paths.all_paths_by_edge:
                exp_all_paths_cost += np.exp(-self.w_ * self.travel_cost_by_one_path(path, d))

        # 计算比值
        if exp_all_paths_cost == 0:
            preference = 0
        else:
            preference = exp_one_path_cost / exp_all_paths_cost  # 这个地方可能0/0，出现nan
        return preference

    def out_road_by_deadline(self,
                             road_e: Edge,
                             vj: Vertex,
                             d):
        '''
        :param road_e: 道路 e
        :param vj: 目的地
        :param d: 车流的截止时间
        :return: 从道路e离开的，目的地是j，截止时间是d的车辆数，求的是最小化的单位
        '''
        now_state = self.state.copy()
        out_vehicles = (now_state[road_e.edge_id, vj.vertex_id, d] * self.single_time) / (
            self.travel_time_on_road(road_e))
        return np.round(out_vehicles)

    def primary_demand(self,
                       vi: Vertex,
                       vj: Vertex,
                       road_e: Edge,
                       d, ):
        primary_vehicles = 0

        if road_e.busy == "normal":  # 随机分配每个路段上会产生的车流量
            primary_vehicles = np.random.poisson(20, 1)
            primary_vehicles = primary_vehicles.sum(axis=0)
        elif road_e.busy == "busy":  # 特殊分配，例如在特定时间段，特定路段上产生大量车流
            pass
        return primary_vehicles

    def secondary_demand(self,
                         vi: Vertex,
                         vj: Vertex,
                         d):
        # 获取终点是i的边的集合
        edge_end_to_vi_list = vi.edges_in_v

        # 对每个顶点，求从该
        sum_demand = 0
        for e in edge_end_to_vi_list:
            sum_demand += self.out_road_by_deadline(e, vj, d)
        return sum_demand

    def in_road_by_deadline(self,
                            road_e: Edge,
                            vj: Vertex,
                            d):
        # 找到road_e为起点的区域
        vertex_start_to_road_e_list = road_e.vertex_in_e

        # 得到从i到j的所有
        primary_demand = 0
        second_demand = 0
        sum_in_road = 0
        preference = 0
        for v in vertex_start_to_road_e_list:
            if v == vj:
                continue
            else:
                all_paths = self.urban.get_path_by_complete_edge(v, vj, p=[])
                for pp in all_paths.all_paths_by_edge:
                    if self.urban.Edges[road_e.edge_id] in pp:
                        primary_demand += self.primary_demand(v, vj, road_e, d)
                        second_demand += self.secondary_demand(v, vj, d)
                        preference = self.path_preference(v, vj, pp, d)
                        sum_in_road += (primary_demand + second_demand) * preference
        return np.round(sum_in_road)

    def step(self,
             action):
        '''
        :param action: Agent执行的动作
        :return: next_state，reward，done，info
        '''
        self.action = action
        next_state = np.zeros((10, 4, 361), dtype=int)
        reward = 0
        done = False
        info = None
        now_state = self.state.copy()

        # 计算下个状态
        for e_key, e_value in self.urban.Edges.items():
            for v_key, v_value in self.urban.Vertexs.items():
                if e_value not in v_value.edges_out_v:
                    for d in range(self.deadline_nums):
                        now_on_road = np.round(now_state[e_key, v_key, d])
                        if now_on_road == 0:
                            in_road = self.in_road_by_deadline(e_value, v_value, d)
                            next_state[e_key, v_key, d] = now_on_road + in_road
                        else:
                            out_road = self.out_road_by_deadline(e_value, v_value, d)
                            # print(out_road)
                            in_road = self.in_road_by_deadline(e_value, v_value, d)
                            next_state[e_key, v_key, d] = now_on_road - out_road + in_road
                        reward += self.reward_arrive_more(e_value, d)

        self.state = next_state.copy()

        # 是否结束

        if self.t == self.step_nums:
            done = True
        self.t += 1  # 转到下个时间步
        # print(next_state.sum(axis=(1,2)))
        # print(action)
        # print("step %d: reward is %d, done is %d" % (self.t-1, reward, done) )

        return self.state.flatten(), reward, done, info

    def reward_beyond_time_least(self,
                                 road_e: Edge,
                                 d):
        r = 0
        if d < (self.t - 1) * self.single_time:
            for v in road_e.vertex_out_e:
                r += self.out_road_by_deadline(road_e, v, d) * (self.t - 1) * self.single_time - d
        return r

    def reward_no_arrive_least(self,
                               road_e: Edge,
                               d):
        r = 0
        if d < (self.t - 1) * self.single_time:
            for v in road_e.vertex_out_e:
                r += self.out_road_by_deadline(road_e, v, d)
        return r

    def reward_arrive_more(self,
                           road_e: Edge,
                           d):
        r = 0
        if d >= (self.t - 1) * self.single_time:
            for v in road_e.vertex_out_e:
                r += self.out_road_by_deadline(road_e, v, d)
        return r
