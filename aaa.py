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
from ogb.graphproppred import DglGraphPropPredDataset
from torch.utils.data import DataLoader

def _collate_fn(batch):
    # 小批次是一个元组(graph, label)列表
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels

# 载入数据集
dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
a, b = dataset[0]
print(a, b)
print(b.dtype)
split_idx = dataset.get_idx_split()
# dataloader
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
