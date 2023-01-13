import matplotlib.pyplot as plt
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from hpc2ml.data.dataset import SimpleDataset
from hpc2ml.data.structuretodata import StructureToData, PAddXArray, PAddPBCEdgeDistance, PAddForce, PAddFracCoords, \
    PAddStress, PAddXPymatgen, PAddSAPymatgen, PAddPBCEdgeXYZ
from hpc2ml.nn.stress import Stress
import unittest

from hpc2ml.data.structuretodata import PAddSAPymatgen, PAddXPymatgen, StructureToData, PAddPBCEdgeXYZ, PAddForce


class MyTestCase(unittest.TestCase):

    def setUp(self):

        addsap = StructureToData(sub_converters=[PAddStress(), PAddPBCEdgeXYZ(), PAddForce()],
                                 n_jobs=1)

        source_path = ["test_data/pure_opt", "test_data/OH_add_static"]
        addsap.sparse_source_data(source_file="vasprun.xml", source_path=source_path, fmt="vasprun",store_path_file=None)
        res = addsap.transform_data_dict()
        print(res)

        self.data = Batch.from_data_list(res)

    def test_stress(self):
        # s = torch.range(-1,1,0.01)
        # threshold = s - torch.floor(s)-0.5
        # plt.plot(s,threshold)
        # plt.show()

        data = self.data
        atom_prop = torch.ones_like(data.pos,requires_grad=True)
        stresslayer = Stress()
        res = stresslayer(atom_prop,data)
        res = res.mean()
        res.backward()


if __name__ == '__main__':
    unittest.main()
