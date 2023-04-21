# import torch as th
# import dgl
# from dgl.data.utils import save_graphs, load_graphs, load_labels
# # u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
# # g = dgl.graph((u, v))
# # graph_labels = {"graph_sizes": th.tensor([3, 3])}
# # print(g)
# # save_path = './gnn_datasets/test1.bin'
# # # save_graphs(save_path, [g], graph_labels)
# # g_list, label_dict = load_graphs(save_path, [0])
# # print(g_list)
# # print(label_dict)
# # node__ = [i for i in range(0, 16)]
# # print(node__)
# a = [1]*10
# a = th.tensor(a)
# print(a)
# 载入OGB的Graph Property Prediction数据集
import dgl
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset, QM7bDataset, DGLDataset
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from dgl.data.utils import save_graphs, load_graphs, load_labels
import numpy as np
import dgl
import torch
from ogb.graphproppred import DglGraphPropPredDataset
from torch.utils.data import DataLoader
from dgl.data.utils import save_graphs, load_graphs, load_labels
import os
import numpy as np
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.conv2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        # self.conv2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.classify = nn.Linear(hidden_dim, n_classes)   # 定义分类器
 
    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 为方便，我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float() # [N, 1]
        h = g.ndata['color'].view(-1, 1).float()
        # 执行图卷积和激活函数
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        # g.ndata['h'] = h    # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')   # [n, hidden_dim]
        return self.classify(hg)  # [n, n_classes]
labels = []
for i in range(100):
    labels.append(i)

labels = np.array(labels)
print(labels)
a = labels.squeeze() % 10
print(a)
b = labels.squeeze()
print(b)
# model = Classifier(1, 16, 10)
# labels = []
# filename = './gnn_datasets/slic_16/test'
# dgl_gs = os.listdir(filename)
# graphs = []
# for file in dgl_gs:
#     file_path = os.path.join(filename, file)
#     graph, label = load_graphs(file_path, [0])
#     graphs.append(graph)
#     # graphs.append(graph)
#     # labels.append(label['class'].item()) 
#     labels.append(label["class"].item())

# trainset = MiniGCDataset(200, 100, 200)  # 生成2000个图，每个图的最小节点数>=10, 最大节点数<=20
# testset = MiniGCDataset(100, 100, 200)
# a = np.unique(labels)
# x = graphs[0][0]
# print(a)
# print(x)
# for i in range(20):
#     print(graphs[i][0])
#     print(graphs[i][0].ndata)
# x,y = testset[0]
# model = Classifier(1, 16, 10)
# pred = model(x)
# print(pred)
# pred = torch.softmax(pred, 1)
# print(pred)
# pred = torch.max(pred, 1)[1].view(-1)
# print(pred)
# print("indegrees:", x.in_degrees())
# print("color", x.ndata['color'])

# def _collate_fn(batch):
#     # 小批次是一个元组(graph, label)列表
#     graphs = [e[0] for e in batch]
#     g = dgl.batch(graphs)
#     labels = [e[1] for e in batch]
#     labels = torch.stack(labels, 0)
#     return g, labels

# # 载入数据集
# dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
# a, b = dataset[0]
# print(a, b)
# print(b.dtype)
# split_idx = dataset.get_idx_split()
# # dataloader
# train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
# valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
# test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
