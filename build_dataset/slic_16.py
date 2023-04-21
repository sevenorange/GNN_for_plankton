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
from skimage import color, io
from skimage.segmentation import find_boundaries, mark_boundaries, slic
from skimage.util import img_as_float


# 将图像数据转换为图形数据
"""
        image--graph
        SLIC分割成为16个超像素区域
        特征: 对应区域的像素平均值
        输入: dataset读入的RGB图像
        输出: dgl格式的graph
"""
def image_to_graph(image):
    # 将图像转换为灰度图像，并转换为 NumPy 数组
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # 转换为灰度图像
    h, w, _ = image.shape
    n_segments = 16
    slic2 = slic(img_as_float(gray_image), n_segments=n_segments, compactness=10)
    n_segments = len(np.unique(slic2))
    m = np.zeros((h, w), dtype="uint8")
    mask = np.array([m] * n_segments)
    for i, t in enumerate(slic2):
        temp = 0
        for j, c in enumerate(t):                
            mask[c-1][i][j] = 255  
    slic_boundary = find_boundaries(slic2, mode='inner').astype(np.uint8) 
    # 创建图形数据
    G = nx.Graph()

    # 将每个超像素区域转换为一个节点，并添加到图中
    colors = []
    for i in np.unique(slic2):
        mask = slic2 == i
        color = np.mean(gray_image[mask], axis=0)
        G.add_node(i, color=color)
        colors.append(color)
    # 通过边界找空间连接关系
    dir = [[0, 1], [0, -1], [1, 0], [1, -1]]
    for i, a in enumerate(slic_boundary):
        for j, b in enumerate(a):
            if b == 1:
                for d in dir:
                    x = i + d[0]
                    y = j + d[1]
                    if x < 0 or x >= h:
                        continue
                    if y < 0 or y >= w:
                        continue
                    left = slic2[x][y]
                    right = slic2[i][j]
                    if left == right:
                        continue
                    if G.has_edge(left, right): continue
                    color_i = G.nodes[left]['color']
                    color_j = G.nodes[right]['color']
                    similarity = np.linalg.norm(color_i - color_j)
                    G.add_edge(left, right, weight=similarity)
                    G.add_edge(left, right, name='aaa') 

    
    
    nodes_count = n_segments     # 节点数量
    nodes_feature = colors    # 节点特征
    # node__ = np.array([arange(0, 4), arange(4, 8), arange(8, 12), arange(12, 16)])

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
save_path = './gnn_datasets/slic_16/'
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
