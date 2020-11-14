# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import matplotlib.pyplot as plt
import numpy as np
from Utils import *
from environments.TrafficEnvironment import *
from Utilities.Config import Config

# Press the green button in the gutter to run the script.

road_matrix = [[0, Edge(0, 0, 1, 4, 1), Edge(1, 0, 2, 5, 1),
                 Edge(2, 0, 3, 3, 1)],
                [Edge(3, 1, 0, 8, 1), 0, 0, Edge(4, 1, 3, 5, 1)],
                [Edge(5, 2, 0, 6, 1), 0, 0, Edge(6, 2, 3, 2, 1)],
                [Edge(7, 3, 0, 7, 1), Edge(8, 3, 1, 4, 1),
                 Edge(9, 3, 2, 12, 1), 0]]

urban = UrbanNetGraph(road_matrix, 4, 10)

config = Config()
#config.road_network = urban
config.environment = TrafficEnvironment(urban)
config.state_size = 4 * 10 * 360
config.action_size = 10
config.num_episodes_to_run = 10000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

if __name__ == '__main__':
    urban = urban.init_infos()
    paths = urban.get_path_by_complete_vertex(urban.Vertexs[0], urban.Vertexs[2], p=[])
    for p in paths:
        for i in p:
            print(i.vertex_id)
        print("\n")
    # config.environment.reset()
    # for i in range(10):
    #     x = np.random.poisson(350, 1)
    #     print(x)
    # x = np.array([[[1,2,3],[4,9,1]],[[1,2,3],[4,5,6]]])
    # x = x.sum(axis=(1,2))
    # print(x)
    # x = x.reshape(x.shape[0], -1)
    # x = torch.tensor(x)
    #
    # x = x.float()
    # net = torch.nn.Linear(6,1200)
    # res = net.forward(x)
    # print(res)

    # for key, value in config.environment.urban.Edges.items():
    #     print(value)
    #     print("\n")
    #     for i in value.vertex_in_e:
    #         print(i.vertex_id)
    # # print(config.environment.urban.Vertexs)
    # x = np.random.randint(0,5,(2,1,3))
    # print(x)
    # print(x.sum(axis=(1,2)))
    #


    # for v in urban.Vertexs:
    #     print(v)
    # for v in urban.Edges:
    #     print(v.edge_capacity)

    # x = np.linspace(1, 50)

    # interval0 = [1 if (i>30) else 0 for i in x]
    # interval1 = [1 if (i<=30) else 0 for i in x]
    #
    # y1 = 6+np.exp(30-x)
    #
    #
    # y = (6+np.exp(30-x))*interval1 + (6+30/x)*interval0
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.plot(x,y1,color='green')
    # plt.figure(1)
    # plt.subplot(1, 2, 2)
    # plt.plot(x,y,color='black')
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
