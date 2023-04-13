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

# 加载图像数据集
dataset_path = './Crustacea/'
train_dir = dataset_path + 'train/'
test_dir = dataset_path + 'test/'
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
])

trainset = ImageFolder(train_dir, transform = transform)
testset = ImageFolder(test_dir, transform = transform)

# 批量转换为graph并存储到本地
graph_save_path = ''
train_graphs_path = graph_save_path + 'train_test.bin'
test_graphs_path = graph_save_path + 'test_t.bin'
for image, label in trainset:
    # image-->graph
    graph, lab = image2graph(image)
# 存储到本地
save_graphs(train_graphs_path, train_graphs, )
