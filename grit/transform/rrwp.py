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
                  max_hash_hops=3,
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

    # Create P_ij
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

    # This is P_ij
    pe = torch.stack(pe_list, dim=-1) # n x n x k

    # Get hashing features
    links = data.edge_index
    hash_dataset = HashDataset(links.to(device), num_nodes, max_hash_hops=max_hash_hops)
    n = num_nodes
    links = torch.from_numpy(np.vstack([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)]).transpose()).to(device)
    hash_pairwise_feature = hash_dataset.elph_hashes.get_bi_subgraph_features(links, hash_dataset.get_hashes(), hash_dataset.get_cards())

    # Unravel back into 3 dimensions
    hash_pairwise_feature = hash_pairwise_feature.reshape(n, n, -1)

    ##### TESTING #####
    # print(hash_pairwise_feature.size())
    # # Check symmetry: print nonsymmetries
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if torch.equal(hash_pairwise_feature[j, i, :], hash_pairwise_feature[i, j, :]) == False:
    #             print(hash_pairwise_feature[j, i, :])
    #             print(hash_pairwise_feature[i, j, :])
    #             print()
    # assert torch.allclose(hash_pairwise_feature, hash_pairwise_feature.transpose(0, 1)), "Features are not symmetric"

    # temp = hash_dataset.elph_hashes.get_bi_subgraph_features(links, hash_dataset.get_hashes(), hash_dataset.get_cards())

    # print(hash_pairwise_feature[0,0,:])
    # print(temp[0])
    # print(hash_pairwise_feature[n-1,n-1,:])
    # print(temp[n*n-1])
    # print(hash_pairwise_feature[0,2,:])
    # print(temp[2])
    ###################

    # Each row (of which there are n) represent a node's features for different powers of the adj matrix P_ij
    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    # Concatenate hash_pairwise_feature into pe
    pe = torch.cat((pe, hash_pairwise_feature), dim=-1)
    # print(f'new pe: {pe.size()}')
                      
    # Convert P_ij into sparse format, extracting the row indices, col indices, and values for non-zero elements
    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    # print(f'dense size: {pe.size()}')

    # Apply Shortest Path Distance (Default false)
    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    # Add attributes to the graph data
    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    

    # print('*'*50)
    # print(data)
    # raise ExceptionError('end of test')
                      
    return data

