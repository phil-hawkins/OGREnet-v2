import torch
from torch.nn import Sequential, Linear, ReLU, Module, ELU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import grid, remove_self_loops
from torch_geometric.data import Data, Batch

edge_attr_sz_0 = 1
node_attr_sz_0 = 4 + 5
u_attr_sz_0 = 256
edge_attr_sz_1 = 512
node_attr_sz_1 = 1
u_attr_sz_1 = 1

edge_h_sz_0 = 1024
node_h_sz_0 = 512

class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        features_in = edge_attr_sz_0 + (2 * node_attr_sz_0) + u_attr_sz_0
        self.edge_mlp = Sequential(
            Linear(features_in, edge_h_sz_0), 
            ReLU(), 
            Linear(edge_h_sz_0, edge_h_sz_0), 
            ReLU(),
            Linear(edge_h_sz_0, edge_h_sz_0), 
            ReLU(),
            Linear(edge_h_sz_0, edge_h_sz_0), 
            ReLU(),
            Linear(edge_h_sz_0, edge_attr_sz_1)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        features_in = edge_attr_sz_1 + node_attr_sz_0
        self.node_mlp_1 = Sequential(
            Linear(features_in, node_h_sz_0), 
            ReLU(), 
            Linear(node_h_sz_0, node_h_sz_0), 
            ReLU()
        )
        features_in = node_attr_sz_0 + node_h_sz_0 + u_attr_sz_0
        self.node_mlp_2 = Sequential(
            Linear(features_in, node_h_sz_0), 
            ReLU(), 
            Linear(node_h_sz_0, node_attr_sz_1)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

# spatial reasoning
class OGRENet(torch.nn.Module):
    def __init__(self):
        super(OGRENet, self).__init__()
        self.select_mlp = Sequential(
            Linear(4096, 256), 
            # ReLU(), 
            # Linear(2048, 1024), 
            # ReLU(), 
            # Linear(1024, 512), 
            # ReLU(), 
            # Linear(512, 256)
        )
        self.g1 = MetaLayer(EdgeModel(), NodeModel(), None) 

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.selection, data.batch
        
        u = self.select_mlp(u)
        x, edge_attr, u = self.g1(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        return torch.squeeze(x)


