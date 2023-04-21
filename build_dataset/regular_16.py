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
        规则分割成4*4
        特征: 对应区域的像素平均值
        输入: dataset读入的RGB图像
        输出: dgl格式的graph
"""
def image_to_graph(image):
    # 将图像转换为灰度图像，并转换为 NumPy 数组
    image = np.array(image)
    h, w, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # 转换为灰度图像
    # 数据标准化
    result = np.zeros(image.shape, dtype=np.float32)
    cv2.normalize(image, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    nodes_count = 4 * 4     # 节点数量
    nodes_feature = np.array([0] * nodes_count, dtype = np.float32)    # 节点特征
    # node__ = np.array([arange(0, 4), arange(4, 8), arange(8, 12), arange(12, 16)])
    nodes__ = np.arange(16).reshape(4, 4)
    lines = np.split(gray_image, 4, axis = 0)
    count_all_node = 0
    for part in lines:
        point = np.split(part, 4, axis = 1)
        for a in point:
            a_feature = np.mean(a)
            nodes_feature[count_all_node] = a_feature
            count_all_node += 1
    h, w = (4, 4)
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
        # G.nodes[i]['color'] = j
    G.add_edges_from(edge_list)

    # dgl_graph = dgl.from_networkx(G, node_attrs=['color'])
    dgl_graph = dgl.from_networkx(G)
    dgl_graph.ndata['color'] = torch.tensor(nodes_feature)
    # print(dgl_graph)
    return dgl_graph
    
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         pass
    
#     def __getitem__(self, index):
#         pass
    
#     def __len__(self):
#         pass

# 加载图像数据集Crustacea
# dataset_path = './Crustacea/'
# train_dir = dataset_path + 'train/'
# test_dir = dataset_path + 'test/'
transform = transforms.Compose([
    # transforms.RandomResizedCrop(size=32),
    transforms.Resize([32, 32]),
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
save_path = './gnn_datasets/regular_16/'
count = 0
for img, label in train_data:
    img_graph = image_to_graph(img)
    # graph_label = {"class": torch.tensor(label)}
    graph_label = {"class": torch.tensor([label], dtype=torch.int16)}
    img_save = save_path + 'train/' + 'train_graph_' + str(count) + '.bin'
    save_graphs(img_save, img_graph, graph_label)
    print(count, label)
    count += 1
count = 0
for img, label in test_data:
    img_graph = image_to_graph(img)
    # graph_label = {"class": torch.tensor(label)}
    graph_label = {"class": torch.tensor([label], dtype=torch.int16)}
    img_save = save_path + 'test/' + 'test_graph_' + str(count) + '.bin'
    save_graphs(img_save, img_graph, graph_label)
    print(count, label)
    count += 1
# test_case = train_data[0]
# test_case_img, test_case_label = test_case
# test_case_label = {"class": torch.tensor(test_case_label)}
# print(test_case_img)
# test_case_graph = image_to_graph(test_case_img)
# save_path = './gnn_datasets/test_case_0.bin'
# save_graphs(save_path, test_case_graph, test_case_label)

# g_list, label_dict = load_graphs(save_path, [0])
# print(g_list)
# print(label_dict)
# print(label_dict['class'])
# 批量转换为graph并存储到本地
# graph_save_path = ''
# train_graphs_path = graph_save_path + 'train_test.bin'
# test_graphs_path = graph_save_path + 'test_t.bin'
# for image, label in trainset:
#     # image-->graph
#     graph, lab = image2graph(image)
# # 存储到本地
# save_graphs(train_graphs_path, train_graphs, )
