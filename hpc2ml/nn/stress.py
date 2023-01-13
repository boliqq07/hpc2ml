import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class Stress(nn.Module):
    """Stress net"""
    def __init__(self, input_dim=3):
        super(Stress, self).__init__()

        self.mlp0 = nn.Sequential(nn.Linear(input_dim, 3), nn.LeakyReLU())

        self.mlp1 = nn.Linear(3, 3, bias=True)
        self.mlp1n = nn.Linear(3, 3, bias=True)

        self.mlp2 = nn.Sequential(nn.Linear(90, 32), nn.ReLU(),
                                  nn.Linear(32, 16), nn.LeakyReLU(),
                                  nn.Linear(16, 6), )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.eye_(self.mlp1.weight)
        nn.init.eye_(self.mlp1n.weight)
        self.mlp1.bias.data.fill_(0)
        self.mlp1n.bias.data.fill_(0)

        for m in self.mlp2:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
                m.weight.data = m.weight.data / 10

    def forward(self, atom_prop, data):
        """
        # atom_prop size (n_atoms, 3)
        # pos (n_atoms, 3)
        # cell (n_sturcture, 3, 3)
        """

        # atom_prop = torch.rand_like(data.pos) # for test

        pos = data.pos
        cell = data.cell
        cell_ravel = cell.view(-1, 9)
        cell_ravel = (cell_ravel.unsqueeze(1) * cell_ravel.unsqueeze(-1)).view(-1, 81)
        cell_ravel = cell_ravel[data.batch]

        cell_1 = torch.inverse(cell)
        frac_pos = torch.bmm(pos.unsqueeze(1), cell_1[data.batch], ).squeeze(1)

        threshold = frac_pos - torch.floor(frac_pos) - 0.5

        # threshold.requires_grad=False

        atom_prop3 = self.mlp0(atom_prop)

        threshold1 = F.relu(self.mlp1(threshold)) * atom_prop3
        threshold2 = F.relu(self.mlp1n(-threshold)) * atom_prop3

        h = torch.cat((atom_prop, threshold1, threshold2, cell_ravel), dim=1)

        h = self.mlp2(h)

        stress = scatter(h, data.batch, dim=0, reduce="sum")

        return stress
