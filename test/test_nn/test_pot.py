# import unittest
#
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch_geometric.data import Batch
# from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter
#
# from hpc2ml.data.structuretodata import PAddStress, PAddXArray, PAddPBCEdgeDistance
# from hpc2ml.data.structuretodata import StructureToData, PAddPBCEdgeXYZ, PAddForce
# from hpc2ml.function.support import add_edge_pbc
# from hpc2ml.nn.activations import Act
# from hpc2ml.nn.lj import LJ
# from hpc2ml.nn.morse import Morse
#
#
# class _PotConv(MessagePassing):
#
#     def __init__(self, nc_edge_hidden=3, pot="morse", mode="xyz", batch_norm=False):
#         super().__init__(aggr='mean')
#         if pot == "morse":
#             self.potlayer = Morse()
#         elif pot == "lj":
#             self.potlayer = LJ()
#         else:
#             raise NotImplementedError
#
#         self.mode = mode
#
#         if self.mode == "xyz":
#             assert nc_edge_hidden == 3  # just for x,y,z vector.
#
#         self.batch_norm = batch_norm
#         if batch_norm:
#             self.bm = nn.BatchNorm1d(nc_edge_hidden)
#
#     def forward(self, edge_index, **kwargs):
#
#         """keep edge_attr is the raw distance_vec, edge_weight is the raw care_ distance"""
#
#
#         size = self.__check_input__(edge_index, size=None)
#
#         coll_dict = self.__collect__(self.__user_args__, edge_index,
#                                      size, kwargs)
#
#         msg_kwargs = self.inspector.distribute('message', coll_dict)
#         for hook in self._message_forward_pre_hooks.values():
#             res = hook(self, (msg_kwargs,))
#             if res is not None:
#                 msg_kwargs = res[0] if isinstance(res, tuple) else res
#
#         out = self.message(**msg_kwargs)
#
#         return out
#
#     def message(self, index, dim_size, z_i, z_j, edge_weight):
#
#         W = self.potlayer(z_i, z_j, edge_weight)
#         if self.batch_norm:
#             W = self.bm(W)
#
#         W_ave = scatter(W, index, dim=0, dim_size=dim_size, reduce= "mean")
#
#         return W, W_ave,z_i, z_j, edge_weight
#
# class MyTestCase(unittest.TestCase):
#
#     def setUp(self):
#         array = pd.read_excel("test_data/single_point_vacuum.xlsx", sheet_name="Sheet2", index_col=0).values
#         addsap = StructureToData(sub_converters=[PAddXArray(array=array),
#                                                  PAddPBCEdgeDistance(cutoff=5.0),
#                                                  # PAddPBCEdgeXYZ(cutoff=5.0),
#                                                  PAddForce(),
#                                                  PAddStress()], n_jobs=1)
#
#         source_path = ["test_data/pure_opt"]
#         addsap.sparse_source_data(source_file="vasprun.xml", source_path=source_path, fmt="vasprun_traj",
#                                   store_path_file=None,space=1)
#         res = addsap.transform_data_dict()
#         print(res)
#
#         self.data = Batch.from_data_list(res)
#
#     def test_converge(self):
#
#         data = self.data
#
#         model = _PotConv()
#
#         model.train()
#
#         from ase.data import covalent_radii
#         radii = torch.from_numpy(covalent_radii).float()
#
#         l = torch.max(torch.unique(data.z))+1
#         pp = torch.rand(l, l) / 10
#         pp = torch.rand(l, l) + pp.T
#
#         def func(center_atom, neighbor_atom, distance,):
#             p1 = pp[center_atom, neighbor_atom]
#             r1 = radii[center_atom]
#             r2 = radii[neighbor_atom]
#             r0 = r1+r2+p1
#             core = -1 * (distance - 1 * r0)
#             pot = 0.4 * torch.pow(1 - torch.exp(core), 2)
#
#             return pot
#
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
#         lm = nn.L1Loss()
#
#         for i in range(100):
#
#             self.optimizer.zero_grad()
#
#             f, f_ave, center_atom, neighbor_atom, distance = model(edge_index=data.edge_index,
#                                                                          edge_weight=data.edge_weight,
#                                                                          edge_attr=data.edge_attr, z=data.z.view(-1, 1))
#             loss = lm(f,func(center_atom, neighbor_atom, distance))
#
#             lossi = loss.mean()
#             print("loop {i}:{lossi.detach().cpu()}")
#             lossi.backward()
#             self.optimizer.step()
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     unittest.main()
#
#
#
#
#
#
#
#
