
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

class OOFPolymerDS(InMemoryDataset):
    def __init__(self, root, target_idx: int, processed_graphs, **kw):
        super().__init__(root, **kw)
        # Use pre-processed graphs with PE applied
        # Check if graphs have multi-target y or single-target y
        if len(processed_graphs) > 0 and processed_graphs[0].y.shape[0] > 1:
            # Multi-target graphs, filter by target_idx
            graphs = [g for g in processed_graphs if torch.isfinite(g.y[target_idx])]
            for g in graphs:
                g.y = g.y[target_idx:target_idx+1]
                g.y_mask = torch.tensor([True])
        else:
            # Single-target graphs (already processed), use as-is
            graphs = processed_graphs
        self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass
