'''
辅助工具类：
路网G
顶点Vertex
边Edge
路径path：

'''
import copy

import numpy as np


class Vertex:
    def __init__(self,
                 vertex_id: int,
                 # vertex_in_degree,
                 # vertex_out_degree
                 ):
        '''
        Vertex: 城市区域类
        :param vertex_id: 区域的编号
        :param vertex_in_degree: 进入该区域的路段数
        :param vertex_out_degree: 从该区域出去的路段数
        '''
        self.vertex_id = vertex_id
        self.edges_in_v = []  # Edge类，进入顶点v的边
        self.edges_out_v = []  # Edge类，从v出去的边
        self.vertex_in_degree = len(self.edges_in_v)
        self.vertex_out_degree = len(self.edges_out_v)

    def get_vertex_id(self):
        '''
        :return: 城市区域编号
        '''
        return self.vertex_id

    def get_vertex_in_degree(self):
        '''
        :return: 城市区域的入度
        '''
        return self.vertex_in_degree

    def get_vertex_out_degree(self):
        '''
        :return: 城市区域的出度
        '''
        return self.vertex_out_degree

    def get_edges_in_v(self):
        '''
        :return:
        '''
        return self.edges_in_v

    def get_edges_out_v(self):
        '''
        :return:
        '''
        return self.edges_out_v


class Edge:
    def __init__(self,
                 edge_id: int,
                 start_point: int,
                 end_point: int,
                 edge_length: int,
                 nums_of_lanes: int,
                 busy="normal"):
        '''
        Edge：道路
        :param edge_id: int. 道路的编号
        :param start_point: int, 道路的起始点
        :param end_point: int, 道路的结束点
        :param edge_length: int, 道路的长度（km）
        :param nums_of_lanes: int, 车道数量
        '''
        self.edge_id = edge_id
        self.start_point = start_point
        self.end_point = end_point
        self.edge_length = edge_length  # 车道长度： km
        self.nums_of_lanes = nums_of_lanes  # 车道数量
        self.edge_capacity = self.edge_length * 500 * self.nums_of_lanes  # 道路的容量，每千米每车道50辆
        self.edge_free_flow_time = self.edge_length * 0.5  # 在该道路上无拥挤行驶时间
        self.vertex_in_e = []  # Vertex类，e的起始点集合
        self.vertex_out_e = []
        self.busy = busy

    def get_edge_id(self):
        '''
        :return: 道路的编号
        '''
        return self.edge_id

    def get_edge_startPoint(self):
        '''
        :return: 道路的起始点
        '''
        return self.start_point

    def get_edge_endPoint(self):
        '''
        :return: 道路的终止点
        '''
        return self.end_point

    def get_edge_length(self):
        '''
        :return: 道路的长度
        '''
        return self.edge_length

    def get_edge_capacity(self):
        '''
        :return: 道路的容量
        '''
        return self.edge_capacity

    def get_edge_free_flow_time(self):
        return self.edge_free_flow_time


class All_Path:
    def __init__(self,
                 PATH_ORI: Vertex,
                 PATH_DEST: Vertex):
        '''
        PATH:从某起点到某终点的路径集合
        :param PATH_origin: 路径起点
        :param PATH_deat: 路径终点
        '''
        self.PATH_ORI = PATH_ORI
        self.PATH_DEST = PATH_DEST
        self.all_paths_by_vertex = []  # [[v1,v2],[v4,v5]], 元素类型是Vertex类
        self.all_paths_by_edge = []  # [[e1,e2],[e4,e5]]，元素类型是Edge类


class path:
    def __init__(self,
                 path_origin: Vertex,
                 path_dest: Vertex):
        '''
        Path：从某起点到某终点的路径
        :param path_origin: 某一条路径的起点
        :param path_dest: 某一条路径的终点
        '''
        self.path_origin = path_origin
        self.path_dest = path_dest
        self.complete_vertexs = []  # =[v1,v2]，里面存的是Vertex类元素
        self.complete_edges = []  # [e1,e2]，里面存的是Edge类元素

    def get_path_origin(self):
        return self.path_origin

    def get_path_dest(self):
        return self.path_dest

    def get_complete_path_by_vertexs(self):
        return self.complete_vertexs

    def get_complete_path_by_edges(self):
        return self.complete_edges


class UrbanNetGraph:
    def __init__(self,
                 real_road_net,
                 vertex_nums: int,
                 edges_nums: int):
        '''
        UrbanNetGraph:城市路网图，计划用邻接矩阵表示，查找是否有边较为方便
        :param real_road_net: 输入的城市路网图
            e.g[0,Edge(1,Vertex(1,2,3),Vertex(2,4,5),3,200),0,edge(2,Vertex(3,1,2),Vertex(4,4,5),6,900)]，0表示无边，Edge则表示有边
        :param vertex_nums: 城市区域总数，
        :param edges_nums: 城市道路总数，元素类型是Edges类
        '''
        self.vertex_nums = vertex_nums
        self.edges_nums = edges_nums
        self.road_net = real_road_net
        # self.Vertexs = set()  # 城市区域集合，元素类型是Vertex类
        # self.Edges = set()  # 道路集合，元素类型是Edge类
        self.Vertexs = {}  # 城市区域集合，元素类型是Vertex类
        self.Edges = {}  # 道路集合，元素类型是Edge类

    def init_infos(self):
        self.generate_RoadNet()
        for key, value in self.Vertexs.items():
            self.get_edges_in_vertex(value)
            self.get_edges_out_vertex(value)
        for key, value in self.Edges.items():
            self.get_vertexs_in_edges(value)
            self.get_vertex_out_edges(value)
        return self

    def generate_RoadNet(self):
        for i in range(self.vertex_nums):
            self.Vertexs[i] = Vertex(i)
            for j in range(self.vertex_nums):
                if self.road_net[i][j] != 0:
                    id = self.road_net[i][j].edge_id
                    self.Edges[id] = self.road_net[i][j]

    def get_vertex_nums(self):
        '''
        :return: 路网中城市区域总数
        '''
        return self.vertex_nums

    def get_edges_nums(self):
        '''
        :return: 路网中道路的总数
        '''
        return self.edges_nums

    def is_valid(self, v: Vertex):
        '''
        :param v: Vertex类
        :return: 判断顶点id是否有效
        '''
        return 0 <= v.vertex_id < self.vertex_nums

    def get_edge_info_by_vertex(self, vi: Vertex, vj: Vertex):
        '''
        :param vi: Vertex类，起点
        :param vj: Vertex类，终点
        :return: 根据顶点的id得到连接两顶点的边的信息
        '''
        if self.is_valid(vi) and self.is_valid(vj):
            return self.road_net[vi.vertex_id][vj.vertex_id]
        else:
            raise ValueError(str(vi.vertex_id) + "or" + str(vj.vertex_id) + "不是有效的顶点")

    def get_edges_in_vertex(self, v: Vertex):
        '''
        :param v: Vertex类
        :return: 道路终点是v的 道路集合
        '''
        for i in range(self.vertex_nums):
            if self.road_net[i][v.vertex_id] != 0:
                v.edges_in_v.append(self.Edges[self.road_net[i][v.vertex_id].edge_id])

    def get_edges_out_vertex(self, v: Vertex):
        '''
        :param v: Vertex类
        :return: 道路起点是v的 道路集合
        '''
        for i in range(self.vertex_nums):
            if self.road_net[v.vertex_id][i] != 0:
                v.edges_out_v.append(self.Edges[self.road_net[v.vertex_id][i].edge_id])

    def get_vertexs_in_edges(self, e: Edge):
        '''
        :param e: Edge类
        :return: e的 开始点集合(Vertex集合）
        '''
        for v in range(self.vertex_nums):
            for vv in range(self.vertex_nums):
                if self.road_net[v][vv] != 0:
                    if self.road_net[v][vv].edge_id == e.edge_id:
                        e.vertex_in_e.append(self.Vertexs[v])
                        break
    def get_vertex_out_edges(self, e:Edge):
        '''
        :param e: Edge类
        :return: e的 结束点集合(Vertex集合）
        '''
        for v in range(self.vertex_nums):
            for vv in range(self.vertex_nums):
                if self.road_net[vv][v] != 0:
                    if self.road_net[vv][v].edge_id == e.edge_id:
                        e.vertex_out_e.append(self.Vertexs[v])
                        break
    # def get_path_by_complete_vertex(self, vi: Vertex, vj: Vertex, p=[]):
    #     '''
    #     :param vi: Vertex类，起点
    #     :param vj: Vertex类，终点
    #     :param p: path类，一条路径
    #     :return: All_Path类，所有路径，计算途径的顶点集合
    #     '''
    #     p.append(self.Vertexs[vi.vertex_id])
    #     # 生成一个所有路径的集合
    #     paths = All_Path(vi.vertex_id, vj.vertex_id)
    #
    #     while(len(p)):
    #         new_p = []
    #         for path in p:
    #             node_row = path[-1]
    #             if node_row == vj:
    #                 paths.all_paths_by_vertex.append(path)
    #                 p.pop(p.index(path))
    #             else:
    #                 adjacent_nodes = np.where(self.road_net[node_row.vertex_id, :])
    #                 adjacent_nodes = adjacent_nodes[0]
    #                 adjacent_nodes = adjacent_nodes.tolist()
    #                 for i in range(len(adjacent_nodes), -1, -1, -1):
    #                     if adjacent_nodes[i] in path:
    #                         adjacent_nodes.pop(i)
    #                     if len(adjacent_nodes) == 0:
    #                         p.pop(p.index(path))
    #                     for node in adjacent_nodes:
    #                         temp = copy.deepcopy(path)
    #                         temp.append(node)
    #                         new_p.append(temp)
    #         p = new_p
    #     return paths



    def get_path_by_complete_vertex(self, vi: Vertex, vj: Vertex, p=[]):
        '''
        :param vi: Vertex类，起点
        :param vj: Vertex类，终点
        :param p: path类，一条路径
        :return: 二维数组，所有路径，计算途径的顶点集合
        '''
        # p.path_origin = vi.vertex_id
        # p.path_dest = vj.vertex_id
        p.append(self.Vertexs[vi.vertex_id])
        # 生成一个所有路径集合
        paths = []
        # paths = All_Path(vi.vertex_id, vj.vertex_id)
        if vi.vertex_id == vj.vertex_id:
            paths.append(p.copy())
            return paths

        for v_key, v_value in self.Vertexs.items():
            if v_value not in p and self.get_edge_info_by_vertex(vi, v_value) != 0:
                new_path = self.get_path_by_complete_vertex(v_value, vj, p)
                for newpath in new_path:
                    paths.append(newpath.copy())
                p.pop()

        return paths

    def get_path_by_complete_edge(self, vi: Vertex, vj: Vertex, p=[]):
        '''
        :param vi: Vertex类，起点
        :param vj: Vertex类，终点集
        :return: All_Path类，所有路径，计算途径的边的合
        '''
        paths = self.get_path_by_complete_vertex(vi, vj, p)
        PATHS = All_Path(vi.vertex_id, vj.vertex_id)
        PATHS.all_paths_by_vertex = paths
        for row in paths:
            road = []
            i = 0
            while i < len(row) - 1:
                road.append(self.Edges[self.road_net[row[i].vertex_id][row[i + 1].vertex_id].edge_id])  # Edge类
                i += 1
            PATHS.all_paths_by_edge.append(road)
        return PATHS
