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
from utils import preprocess_adj
# from torch_geometric.utils import to_dense_batch, to_dense

# class 
# 定义 GNN 模型
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        # self.pre = image_to_graph()
        self.conv1 = GCNConv(3, 16)  # 输入特征维度为 3，输出特征维度为 16
        self.conv2 = GCNConv(16, 32) # 输入特征维度为 16，输出特征维度为 32
        self.conv3 = GCNConv(32, 2)  # 输入特征维度为 32，输出特征维度为 2
        self.relu = nn.ReLU()

    
    
# class I2G(nn.Module):
#     def __init__(self):
#         super(GNNModel, self).__init__()
    # 将图像数据转换为图形数据
    """
        image--graph
        规则分割
        输入: dataset读入的RGB图像
        输出: 邻接矩阵(edge_index)和节点特征(features)的tensor
    """
    def image_to_graph(self, image):
        # 将图像转换为灰度图像，并转换为 NumPy 数组
        print('image格式:', image.dtype)
        # image = image.cpu().numpy() 
        # image = image[0]
        print('image维度', image.shape)
        image = image.permute(1, 2, 0)
        image = np.transpose(image,(1,2,0)) # 输入维度为CHW, 转换为HWC
        h, w, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # 转换为灰度图像
        
        nodes_count = h * w     # 节点数量
        nodes__ = np.zeros((h, w), dtype = np.int16)    # 
        count = int(0)
        for i, line in enumerate(gray_image):
            for j, _ in enumerate(line):
                nodes__[i][j] = count
                count += 1
        nodes_feature = np.array([0] * nodes_count, dtype = np.float32)    # 节点特征
        print(len(nodes_feature))   # 节点数 1024
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
        print(len(edge_list))

        # 构造图
        G = nx.Graph()
        for i, j in enumerate(nodes_feature):
            G.add_node(i)
            G.nodes[i]['color'] = j
        G.add_edges_from(edge_list)
        # adj = nx.adjacency_matrix(G).todense()
        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        adj_t = preprocess_adj(adj)
        print(adj_t)

        dgl_graph = dgl.from_networkx(G, node_attrs=['color'])
        print(dgl_graph)

        # 生成节点特征tensor
        row = list(range(nodes_count))
        col = list(range(nodes_count))
        indices = torch.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = torch.tensor(nodes_feature)
        features = torch.sparse.FloatTensor(indices, values,
                                                (nodes_count, nodes_count))
        print(features)
        print(features.dtype)
        return features, adj_t 

    def forward(self, data):
        x, edge_index = self.image_to_graph(data)
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        return x

# 加载数据并转换为图形表示
transform = transforms.Compose(
    # [transforms.Resize((32, 32)),
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #  torchvision.transforms.Normalize((0.1307,), (0.3081,))
     ])

# 加载图像数据
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)


    # 将边缘转换为图形表示
    # edge_list = []
    # for i in range(edges.shape[0]):
    #     for j in range(edges.shape[1]):
    #         if edges[i,j] == 255:
    #             edge_list.append((i,j))

    # 将图形表示转换为 PyTorch Geometric 数据格式
    # edge_index = torch.LongTensor(list(zip(*edge_list)))
    # x = torch.FloatTensor(gray_image.reshape(-1, 1))
    # data = Data(x=x, edge_index=edge_index)
    # data = to_dense(data)
    # return data

# train_graphs = [image, label for image, label in train_data]
# test_graphs = [image, label for image, label in test_data]
# train_graphs = [image_to_graph(image) for image, label in train_data]
# test_graphs = [image_to_graph(image) for image, label in test_data]
# train_graphs = DataLoader(train_data, batch_size=16)
# test_graphs = DataLoader(test_data, batch_size=16)



# 定义训练函数
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# 定义测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            total_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = total_loss / len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

# 训练 GNN 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

for epoch in range(10):
    train(model, device, train_loader, optimizer, criterion)
    test(model, device, test_loader, criterion)
