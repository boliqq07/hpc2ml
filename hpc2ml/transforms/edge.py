
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops

from hpc2ml.function.support import add_edge_no_pbc, add_edge_no_pbc_from_index, add_edge_pbc_from_index, add_edge_pbc, \
    distribute_edge


class AddEdgeNoPBC(BaseTransform):
    """Add edge index, with no pbc."""

    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, data):
        if hasattr(data, "edge_index"):
            data = add_edge_no_pbc_from_index(data)
        else:
            data = add_edge_no_pbc(data, cutoff=self.cutoff)
        return data


class AddEdgePBC(BaseTransform):
    """Add edge index, with no pbc."""

    def __init__(self, cutoff=5.0, max_neighbors=16):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def __call__(self, data):
        if hasattr(data, "edge_index"):
            data = add_edge_pbc_from_index(data)
        else:
            data = add_edge_pbc(data, cutoff=self.cutoff, max_neighbors=self.max_neighbors)
        return data


class AddAttrToWeight(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)
        return data


class AddAttrSumToAttrAndWeight(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)
            data.edge_attr = data.edge_weight.reshape(-1, 1)
        return data


class AddAttrSumToAttr(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        assert hasattr(data, "edge_attr") and data.edge_attr is not None
        assert data.edge_attr.shape[1] == 3
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)

        wei = data.edge_weight

        data.edge_attr = torch.cat((wei.reshape(-1, 1), data.edge_attr), dim=1)

        return data


class DistributeEdgeAttr(BaseTransform):
    """Compact with rotnet network (deep potential)"""

    def __init__(self, r_cs=2.0, r_c=6.0, cat_weight_attr=True):
        super().__init__()
        self.r_cs = r_cs
        self.r_c = r_c
        self.cat_weight_attr = cat_weight_attr

    def __call__(self, data):
        data = distribute_edge(data, self.r_cs, self.r_c)
        return data