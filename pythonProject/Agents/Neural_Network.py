import torch
import torch.nn as nn
import torch.nn.functional as F



class NerualNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NerualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.input_dim, 1200)
        self.linear2 = nn.Linear(1200, 360)
        self.linear3 = nn.Linear(360, self.output_dim)


        # 搭建神经网络

    # 进行前向传播
    def forward(self, x):
        # 把x的后两维拉成一列
        x = x.reshape(x.shape[0], -1)
        x = x.float()
        # x = torch.from_numpy(x) # Tensor
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
        



