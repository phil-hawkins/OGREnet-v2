import torch
from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU, Module, ELU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import grid, remove_self_loops
from torch_geometric.data import Data, Batch

# edge_attr_sz_0 = 1
# node_attr_sz_0 = 4 + 5
# u_attr_sz_0 = 256
# edge_attr_sz_1 = 512
# node_attr_sz_1 = 1
# u_attr_sz_1 = 1

# edge_h_sz_0 = 1024
# node_h_sz_0 = 512

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_attr_sz_in, edge_attr_sz_out, edge_h_sz, node_attr_sz, u_attr_sz, hidden_layers=3):
        super(EdgeModel, self).__init__()
        features_in = edge_attr_sz_in + (2 * node_attr_sz) + u_attr_sz

        module_odict = [
            ('edge linear 0', Linear(features_in, edge_h_sz)), 
            ('edge relu 0', ReLU())
        ]

        for i in range(hidden_layers):
            module_odict.append(('edge linear {}'.format(i+1), Linear(edge_h_sz, edge_h_sz)))
            module_odict.append(('edge relu {}'.format(i+1), ReLU()))

        module_odict.append(('edge linear final', Linear(edge_h_sz, edge_attr_sz_out)))
        self.edge_mlp = Sequential(OrderedDict(module_odict))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_attr_sz_in, node_attr_sz_out, edge_attr_sz, node_h_sz, u_attr_sz, hidden_layers=1):
        super(NodeModel, self).__init__()
        features_in = node_attr_sz_in + edge_attr_sz

        module_odict = [
            ('node linear 0', Linear(features_in, node_h_sz)), 
            ('node relu 0', ReLU())
        ]

        for i in range(hidden_layers):
            module_odict.append(('node linear {}'.format(i+1), Linear(node_h_sz, node_h_sz)))
            module_odict.append(('node relu {}'.format(i+1), ReLU()))

        self.node_mlp_1 = Sequential(OrderedDict(module_odict))

        features_in = node_attr_sz_in + node_h_sz + u_attr_sz
        self.node_mlp_2 = Sequential(
            Linear(features_in, node_h_sz), 
            ReLU(), 
            Linear(node_h_sz, node_attr_sz_out)
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
    def __init__(self, u_attr_sz=4096, u_attr_reduced_sz=256, edge_h_sz=1024, edge_attr_sz1=512, node_h_sz=512, edge_hidden_layers=3, node_hidden_layers=1):
        super(OGRENet, self).__init__()
        node_attr_sz_in = 4 + 5
        self.select_dim_reduction = Linear(u_attr_sz, u_attr_reduced_sz)
        self.g1 = MetaLayer(
            EdgeModel(
                edge_attr_sz_in=1, 
                edge_h_sz=edge_h_sz, 
                edge_attr_sz_out=edge_attr_sz1, 
                node_attr_sz=node_attr_sz_in, 
                u_attr_sz=u_attr_reduced_sz, 
                hidden_layers=edge_hidden_layers
            ), 
            NodeModel(
                node_attr_sz_in=node_attr_sz_in, 
                node_attr_sz_out=1, 
                edge_attr_sz=edge_attr_sz1, 
                node_h_sz=node_h_sz, 
                u_attr_sz=u_attr_reduced_sz, 
                hidden_layers=node_hidden_layers), 
            None
        ) 

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.selection, data.batch
        
        u = self.select_dim_reduction(u)
        x, edge_attr, u = self.g1(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        return torch.squeeze(x)


