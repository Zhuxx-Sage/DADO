# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import matplotlib.pyplot as plt
import numpy as np
from Utils import *

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    road_graph = [[0, Edge(0, Vertex(0),Vertex(1), 4, 1), Edge(1, Vertex(0), Vertex(2), 5, 1), Edge(2, Vertex(0), Vertex(3), 3, 1)],
                  [Edge(3, Vertex(1), Vertex(0), 8, 1), 0, 0, Edge(4, Vertex(1), Vertex(3), 5, 1)],
                  [Edge(5, Vertex(2), Vertex(0), 6, 1), 0, 0, Edge(6, Vertex(2), Vertex(3), 2, 1)],
                  [Edge(7, Vertex(3), Vertex(0), 7, 1), Edge(8, Vertex(3), Vertex(1), 4, 1), Edge(9, Vertex(3), Vertex(2), 12, 1), 0]]

    urban = UrbanNetGraph(road_graph, 4, 10)

                     urban.init_infos()
    for v in urban.Vertexs:
        print(v.vertex_id)
    for v in urban.Edges:
        print(v.edge_id)

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
