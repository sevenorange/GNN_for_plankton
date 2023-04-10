import torch
import torchvision
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import torchvision.transforms as transforms
from torch_geometric.data import Data
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
from torch_geometric.utils import to_dense_batch, to_dense

# 定义 GNN 模型
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(3, 16)  # 输入特征维度为 3，输出特征维度为 16
        self.conv2 = GCNConv(16, 32) # 输入特征维度为 16，输出特征维度为 32
        self.conv3 = GCNConv(32, 2)  # 输入特征维度为 32，输出特征维度为 2
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        return x
