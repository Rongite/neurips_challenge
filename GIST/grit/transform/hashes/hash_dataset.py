import torch
from time import time
from torch_geometric.data import Dataset
import scipy.sparse as ssp

from .hashing import ElphHashes
from .utils import get_src_dst_degree

class HashDataset():
    """
    A class that combines propagated node features x, and subgraph features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, edge_index, num_nodes, max_hash_hops=3, hll_p=12, minhash_num_perm=256, **kwargs):
        self.max_hash_hops = max_hash_hops
        self.hll_p = hll_p
        self.elph_hashes = ElphHashes(max_hash_hops=self.max_hash_hops, hll_p=self.hll_p, minhash_num_perm=minhash_num_perm)  # object for hash and subgraph feature operations
        self.subgraph_features = None
        self.hashes = None
        self.device = edge_index.device
        super(HashDataset, self).__init__()

        self.links = self.edge_index = edge_index
        self.labels = None
        self.edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0].to('cpu'), self.edge_index[1].to('cpu'))),
            shape=(num_nodes, num_nodes)
        )
        self.edge_index[0].to(self.device)
        self.edge_index[1].to(self.device)
        self.edge_index.to(self.device)

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()
        # pre-process subgraph features
        self._preprocess_subgraph_features(num_nodes)

    def _preprocess_subgraph_features(self, num_nodes):
        """
        Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
        Sets self.subgraph_features
        @return:
        """
        # print('generating subgraph features')
        start_time = time()
        self.hashes, self.cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
        # print("Preprocessed hashes in: {:.2f} seconds".format(time() - start_time))

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            subgraph_features = self.subgraph_features[idx]
        else:
            subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        if self.use_RA:
            RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
            RA = torch.tensor([RA], dtype=torch.float)
        else:
            RA = -1
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return subgraph_features, node_features, src_degree, dst_degree, RA, y

    def get_hashes(self):
        return self.hashes

    def get_cards(self):
        return self.cards
