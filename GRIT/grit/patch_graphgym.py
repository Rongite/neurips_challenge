
# auto-generated patch so GraphGym <0.3 accepts 5-arg train()
import torch_geometric.graphgym.train as _t
def _train_compat(*args, **kw):
    return _t.train(*(args[:4]), **kw) if len(args) == 5 else _t.train(*args, **kw)
_t.train = _train_compat
