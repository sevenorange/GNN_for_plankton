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
import dgl
from timm.data import Dataset
from torchvision.datasets import ImageFolder
from dgl.data.utils import save_graphs, load_graphs, load_labels

# 将图像数据转换为图形数据
"""
        image--graph
        slic分割
        特征: 平均像素特征
        输入: dataset读入的RGB图像
        输出: 邻接矩阵(edge_index)和节点特征(features)的tensor
"""
def image_to_graph(image):
    # 将图像转换为灰度图像，并转换为 NumPy 数组
    # print('image格式:', image.dtype)
    # image = image.cpu().numpy() 
    # image = image[0]
    image = np.array(image)
    # print('image维度', image.shape)
    # image = image.permute(1, 2, 0)
    # image = np.transpose(image,(1,2,0)) # 输入维度为CHW, 转换为HWC
    h, w, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # 转换为灰度图像

    node__ = [i for i in range(0, 16)]
    lines = np.split(gray_image, 4, axis = 0)
    count_all_node = 0
    for part in lines:
        point = np.split(part, 4, axis = 1)
        for a in point:

        
    nodes_count = h * w     # 节点数量
    nodes__ = np.zeros((h, w), dtype = np.int16)    # 
    count = int(0)
    for i, line in enumerate(gray_image):
        for j, _ in enumerate(line):
            nodes__[i][j] = count
            count += 1
    nodes_feature = np.array([0] * nodes_count, dtype = np.float32)    # 节点特征
    # print(len(nodes_feature))   # 节点数 1024
    edge_list = []  # 边列表
    for i, line in enumerate(nodes__):
        for j, node_index in enumerate(line):
            # 所有相邻的节点间生成1条边, 由上向下, 由左向右
            nodes_feature[node_index] = gray_image[i][j]
            if i < h-1:
                edge_list.append((node_index, nodes__[i+1][j]))
                if j != 0:
                    edge_list.append((node_index, nodes__[i+1][j-1]))
                if j < w-1:
                    edge_list.append((node_index, nodes__[i+1][j+1]))
            if j < w-1:
                edge_list.append((node_index, nodes__[i][j+1]))
    # print(len(edge_list))

    # 构造图
    G = nx.Graph()
    for i, j in enumerate(nodes_feature):
        G.add_node(i)
        G.nodes[i]['color'] = j
    G.add_edges_from(edge_list)

    dgl_graph = dgl.from_networkx(G, node_attrs=['color'])
    # print(dgl_graph)
    return dgl_graph
    

# 加载图像数据集Crustacea
# dataset_path = './Crustacea/'
# train_dir = dataset_path + 'train/'
# test_dir = dataset_path + 'test/'
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
])

# trainset = ImageFolder(train_dir, transform = transform)
# testset = ImageFolder(test_dir, transform = transform)

# 加载图像数据CIFAR10
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
# 存储图数据集到本地
save_path = './gnn_datasets/regular_grid/'
count = 0
for img, label in train_data:
    img_graph = image_to_graph(img)
    graph_label = {"class": torch.tensor(label)}
    img_save = save_path + 'train/' + 'train_graph_' + str(count) + '.bin'
    save_graphs(img_save, img_graph, graph_label)
    count += 1
count = 0
for img, label in test_data:
    img_graph = image_to_graph(img)
    graph_label = {"class": torch.tensor(label)}
    img_save = save_path + 'test/' + 'test_graph_' + str(count) + '.bin'
    save_graphs(img_save, img_graph, graph_label)
    count += 1
