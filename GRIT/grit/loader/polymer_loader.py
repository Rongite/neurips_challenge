
from pathlib import Path
import torch, random
from torch_geometric.data import InMemoryDataset

class PolymerDS(InMemoryDataset):
    def __init__(self, root, target_idx: int, **kw):
        super().__init__(root, **kw)
        graphs = torch.load(Path(root) / "train_supplement" / "graphs" / "train_graphs.pt", map_location='cpu', weights_only=False)
        graphs = [g for g in graphs if g.y_mask[target_idx]]
        for g in graphs:
            g.y = g.y[target_idx:target_idx+1]
            g.y_mask = torch.tensor([True])
        self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass
