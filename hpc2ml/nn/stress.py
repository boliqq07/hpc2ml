import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_scatter import scatter


class Stress(nn.Module):
    """Stress net"""
    def __init__(self, input_dim=3,readout="mean"):
        super(Stress, self).__init__()

        self.mlp0 = nn.Sequential(nn.Linear(input_dim + 3 + 81, 128, bias=True), nn.ReLU())

        self.bm=nn.BatchNorm1d(input_dim + 3 + 81)

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64))

        self.readout = readout

        if self.readout == "set2set":
            self.set2set = aggr.Set2Set(64, processing_steps=2)

            self.mlp3 = nn.Sequential(
                nn.Linear(64*2, 32), nn.ReLU(),
                nn.Linear(32, 6))
        else:
            self.mlp3 = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 6))

    def forward(self, atom_prop, data):
        """
        # atom_prop size (n_atoms, 3)
        # pos (n_atoms, 3)
        # cell (n_sturcture, 3, 3)
        """

        pos = data.pos
        cell = data.cell
        cell_ravel = cell.view(-1, 9)
        cell_ravel = (cell_ravel.unsqueeze(1) * cell_ravel.unsqueeze(-1)).view(-1, 81)
        cell_ravel = cell_ravel[data.batch]

        if not hasattr(data, "frac_pos"):
            cell_1 = torch.inverse(cell)
            frac_pos = torch.bmm(pos.unsqueeze(1), cell_1[data.batch], ).squeeze(1)
        else:
            frac_pos = data.frac_pos

        h = torch.cat((atom_prop, frac_pos, cell_ravel), dim=1)

        h = self.bm(h)
        h = self.mlp0(h)
        h = self.mlp2(h)

        if self.readout == "set2set":
            h = self.set2set(h, data.batch)
            stress = self.mlp3(h)
        else:
            h = scatter(h, data.batch, dim=0, reduce=self.readout)
            stress = self.mlp3(h)

        return stress


# class Stress(nn.Module):
#     """Stress net"""
#     def __init__(self, input_dim=3):
#         super(Stress, self).__init__()
#
#         self.mlp0 = nn.Linear(input_dim, 32)
#
#         self.mlp1 = nn.Linear(3, 3, bias=True)
#         self.mlp1n = nn.Linear(3, 3, bias=True)
#
#         self.mlp2 = nn.Sequential(nn.Linear(87+32, 32), nn.ReLU(),
#                                   nn.Linear(32, 16), nn.ReLU(),
#                                   nn.Linear(16, 6), )
#         self.reset_parameters()
#         self.readout = "set2set"
#         self.readout = "mean"
#
#         if self.readout == "set2set":
#             self.set2set = aggr.Set2Set(6, processing_steps=2)
#             self.lin1 = torch.nn.Linear(2 * 6, 6)
#         else:
#             self.lin1 = torch.nn.Linear(6, 6)
#
#     def reset_parameters(self) -> None:
#         nn.init.eye_(self.mlp1.weight)
#         nn.init.eye_(self.mlp1n.weight)
#         self.mlp1.bias.data.fill_(0)
#         self.mlp1n.bias.data.fill_(0)
#
#         for m in self.mlp2:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0)
#
#     def forward(self, atom_prop, data):
#         """
#         # atom_prop size (n_atoms, 3)
#         # pos (n_atoms, 3)
#         # cell (n_sturcture, 3, 3)
#         """
#
#         # atom_prop = torch.rand_like(data.pos) # for test
#
#         pos = data.pos
#         cell = data.cell
#         cell_ravel = cell.view(-1, 9)
#         cell_ravel = (cell_ravel.unsqueeze(1) * cell_ravel.unsqueeze(-1)).view(-1, 81)
#         cell_ravel = cell_ravel[data.batch]
#
#         if not hasattr(data,"frac_pos"):
#             cell_1 = torch.inverse(cell)
#             frac_pos = torch.bmm(pos.unsqueeze(1), cell_1[data.batch], ).squeeze(1)
#         else:
#             frac_pos = data.frac_pos
#
#         threshold = frac_pos - torch.floor(frac_pos) - 0.5
#
#         # threshold.requires_grad=False
#
#         atom_prop = self.mlp0(atom_prop)
#
#         threshold1 = F.relu(self.mlp1(threshold))
#         threshold2 = F.relu(self.mlp1n(-threshold))
#
#         h = torch.cat((atom_prop, threshold1, threshold2, cell_ravel), dim=1)
#
#         h = self.mlp2(h)
#
#         if self.readout == "set2set":
#             h = F.relu(self.set2set(h, data.batch))
#
#         else:
#             h = F.relu(scatter(h, data.batch, dim=0, reduce=self.readout))
#
#         stress = self.lin1(h)
#
#         return stress
