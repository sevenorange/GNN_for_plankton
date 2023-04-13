import torch as th
import dgl
from dgl.data.utils import save_graphs, load_graphs, load_labels
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
graph_labels = {"graph_sizes": th.tensor([3, 3])}
print(g)
save_path = './gnn_datasets/test1.bin'
# save_graphs(save_path, [g], graph_labels)
g_list, label_dict = load_graphs(save_path, [0])
print(g_list)
print(label_dict)