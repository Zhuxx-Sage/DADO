'''
辅助工具类：
路网G
顶点Vertex
边Edge
路径path：

'''


class Vertex:
    def __init__(self,
                 vertex_id,
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
                 edge_id,
                 start_point,
                 end_point,
                 edge_length,
                 nums_of_lanes):
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
        self.edge_capacity = self.edge_length * 50 * self.nums_of_lanes  # 道路的容量，每千米每车道50辆
        self.edge_free_flow_time = self.edge_length * 0.5  # 在该道路上无拥挤行驶时间
        self.vertex_in_e = []  # Vertex类，进入e的顶点集合

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
                 vertex_nums,
                 edges_nums):
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
        self.Vertexs = set()  # 城市区域集合，元素类型是Vertex类
        self.Edges = set()  # 道路集合，元素类型是Edge类

    def init_infos(self):
        self.generate_RoadNet()
        for v in self.Vertexs:
            self.get_edges_in_vertex(v)
            self.get_edges_out_vertex(v)

        for e in self.Edges:
            self.get_vertexs_in_edges(e)


    def generate_RoadNet(self):
        for i in range(self.vertex_nums):
            for j in range(self.vertex_nums):
                if self.road_net[i][j] != 0:
                    self.Edges.add(self.road_net[i][j])
                    self.Vertexs.add(Vertex(self.road_net[i][j].start_point))

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
                v.edges_in_v.append(self.road_net[i][v.vertex_id])

    def get_edges_out_vertex(self, v: Vertex):
        '''
        :param v: Vertex类
        :return: 道路起点是v的 道路集合
        '''
        for i in range(self.vertex_nums):
            if self.road_net[v.vertex_id][i] != 0:
                v.edges_out_v.append(self.road_net[v.vertex_id][i])

    def get_vertexs_in_edges(self, e: Edge):
        '''
        :param e: Edge类
        :return: 进入e的 区域集合
        '''
        for v in range(self.vertex_nums):
            for vv in range(self.vertex_nums):
                if self.road_net[v][vv] == e:
                    e.vertex_in_e.append(v)

    def get_path_by_complete_vertex(self, vi: Vertex, vj: Vertex, p: path):
        '''
        :param vi: Vertex类，起点
        :param vj: Vertex类，终点
        :param p: path类，一条路径
        :return: All_Path类，所有路径，计算途径的顶点集合
        '''
        p.path_origin = vi.vertex_id
        p.path_dest = vj.vertex_id
        p.complete_vertexs.append(vi)
        # 生成一个所有路径集合
        paths = All_Path(vi.vertex_id, vj.vertex_id)
        if vi.vertex_id == vj.vertex_id:
            paths.all_paths_by_vertex.append(p)
            return paths

        for v in self.Vertexs:
            if v not in p.complete_vertexs and self.get_edge_info_by_vertex(vi, v) != 0:
                new_path = self.get_path_by_complete_vertex(v, vj, p)
                for newpath in new_path:
                    paths.all_paths_by_vertex.append(newpath)

        return paths

    def get_path_by_complete_edge(self, vi: Vertex, vj: Vertex, p: path):
        '''
        :param vi: Vertex类，起点
        :param vj: Vertex类，终点
        :return: All_Path类，所有路径，计算途径的边的集合
        '''
        paths = self.get_path_by_complete_vertex(vi, vj, p).all_paths_by_vertex
        for row in paths:
            i = 0
            while i < len(row)-1:
                p.complete_edges.append(self.road_net[row[i].id][row[i+1].id])
                i += 1
            paths.all_paths_by_edge.append(p.complete_edges)
        return paths

