# import numpy as np
# from torchvision import transforms
# import torch
# import cv2

# # a = np.random.random((224,224,3))
# a = np.random.randint(255, size=(224, 224, 3))
# a = cv2.imread('./GNN_test/Lenna.png')
# a = np.array(a, dtype=np.float32)
# # a = np.transpose(a, (2, 0, 1))
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# b = transform(a)
# print(type(b))
# print(b.dtype)
# print(b.shape)
# # print(b)
import torch as th
import dgl
import matplotlib.pyplot as plt
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
u1, v1 = th.tensor([0, 1, 2, 3]), th.tensor([2, 3, 4, 4])
u2, v2 = th.tensor([0, 0, 2, 3]), th.tensor([2, 3, 3, 1])
g = dgl.graph((u, v))
g1 = dgl.graph((u1, v1))
g2 = dgl.graph((u2, v2))
# 每个节点赋值特征
g.ndata['x'] = th.randn(g.num_nodes(), 3)  # 长度为3的节点特征
g.ndata['mask'] = th.randn(g.num_nodes(), 3)  # 节点可以同时拥有不同方面的特征
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32) # 每个边赋值特征
g2.ndata['x'] = th.randn(g2.num_nodes(), 3)  # 长度为3的节点特征
g2.ndata['mask'] = th.randn(g2.num_nodes(), 3)  # 节点可以同时拥有不同方面的特征
g2.edata['x'] = th.ones(g2.num_edges(), dtype=th.int32) # 每个边赋值特征
g1.ndata['x'] = th.randn(g1.num_nodes(), 3)  # 长度为3的节点特征
g1.ndata['mask'] = th.randn(g1.num_nodes(), 3)  # 节点可以同时拥有不同方面的特征
g1.edata['x'] = th.ones(g1.num_edges(), dtype=th.int32) # 每个边赋值特征
print(g)

print(g.ndata["x"][0])
print(g.edata["x"][0])

# print(g)
import networkx as nx
# %matplotlib inline
graphs = dgl.batch([g, g1, g2])

# nx_G = g.to_networkx()
nx_G = graphs.to_networkx()
# # 可视化图形数据
fig = plt.figure('Graphs')
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.axis('equal')
plt.show()

