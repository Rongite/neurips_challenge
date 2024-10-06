# ------------------------ : new rwpse ----------------
from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter, scatter_add, scatter_max

from torch_geometric.graphgym.config import cfg

from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)
import torch_sparse
from torch_sparse import SparseTensor

from .hashes.hash_dataset import HashDataset


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data



@torch.no_grad()
def add_full_rrwp(data,
                  walk_length=8,
                  attr_name_abs="rrwp", # name: 'rrwp'
                  attr_name_rel="rrwp", # name: ('rrwp_idx', 'rrwp_val')
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes),
                                       )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    # HERE       

    #####
    max_hash_hops = 2
    N_nodes_max = 444
    #####
    links = data.edge_index
    
    hash_dataset = HashDataset(links.to(device), num_nodes, max_hash_hops=max_hash_hops)
    n = num_nodes
    links = torch.from_numpy(np.vstack([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)]).transpose()).to(device)
    hash_pairwise_feature = hash_dataset.elph_hashes.get_bi_subgraph_features(links, hash_dataset.get_hashes(), hash_dataset.get_cards()).detach().cpu().numpy()
    # Pad hash features
    # Initialize the new array with zeros
    padded_hash_pairwise_feature = np.zeros((N_nodes_max ** 2, hash_pairwise_feature.shape[1]))
    # Fill the padded array
    current_position = 0
    for i in range(0, hash_pairwise_feature.shape[0], num_nodes):
        padded_hash_pairwise_feature[current_position:current_position + num_nodes] = hash_pairwise_feature[i:i + num_nodes]
        current_position += N_nodes_max
    
    data.hash_pairwise_feature = torch.from_numpy(padded_hash_pairwise_feature).to(torch.float32)
    data.number_features = len(data.x[0])        
    
    # Get adjacency matrix
    adj_matrix = np.zeros((N_nodes_max, N_nodes_max, data.edge_attr.shape[1]))
    
    for i,j,attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        adj_matrix[i][j] = (attr + 1).cpu().numpy()
        adj_matrix[j][i] = (attr + 1).cpu().numpy()    # have to add this because edge_index is undirected
    data.edge_feature_dim = len(adj_matrix[0][0])
    data.adj_matrix = torch.from_numpy(adj_matrix).to(torch.float32)

    print('*'*50)
    print(data)
    raise ExceptionError('end of test')
                      
    return data

