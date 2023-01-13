import unittest

import pandas as pd
import torch
from torch_geometric.data import Batch

from hpc2ml.data.structuretodata import PAddStress, PAddXArray, PAddPBCEdgeDistance
from hpc2ml.data.structuretodata import StructureToData, PAddPBCEdgeXYZ, PAddForce
from hpc2ml.nn.cggruforcestress import CGGRUForceStress


class MyTestCase(unittest.TestCase):

    def setUp(self):
        array = pd.read_excel("test_data/single_point_vacuum.xlsx", sheet_name="Sheet2", index_col=0).values
        addsap = StructureToData(sub_converters=[PAddXArray(array=array),
                                                 PAddPBCEdgeDistance(cutoff=5.0),
                                                 # PAddPBCEdgeXYZ(),
                                                 PAddForce(),
                                                 PAddStress()], n_jobs=1)

        source_path = ["test_data/pure_opt", "test_data/OH_add_static"]
        addsap.sparse_source_data(source_file="vasprun.xml", source_path=source_path, fmt="vasprun",store_path_file=None)
        res = addsap.transform_data_dict()
        print(res)

        self.data = Batch.from_data_list(res)

    def test_energy(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=1, dim=16, cutoff=6.0, n_block=3, get_force=False,)
        res = stresslayer(data)
        res = res.mean()
        res.backward()

    def test_force_direct(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=1, dim=16, cutoff=6.0, n_block=3,
                                       get_force=True,direct_force=True,
                                       get_stress=False)
        stresslayer.to("cuda:0")
        data.to("cuda:0")

        res = stresslayer(data)
        res = res[0].mean()+res[1].mean()
        res.backward()

    def test_force_no_direct(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=1, dim=16, cutoff=6.0, n_block=2,
                                       get_force=True, direct_force=False,try_add_edge_msg=True,
                                       get_stress=False)
        stresslayer.to("cpu")
        data.to("cpu")

        res = stresslayer(data)
        res = res[0].mean()+res[1].mean()
        res.backward()

    def test_force_no_direct_stress(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=1, dim=16, cutoff=6.0, n_block=2,
                                       get_force=True, direct_force=False, try_add_edge_msg=True,
                                       get_stress=True)
        stresslayer.to("cpu")
        data.to("cpu")

        res = stresslayer(data)
        res = res[0].mean()+res[1].mean()+res[2].mean()
        res.backward()

    def test_force_direct_stress(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=1, dim=16, cutoff=6.0, n_block=2,
                                       get_force=True, direct_force=True,
                                       get_stress=True)
        stresslayer.to("cuda:0")
        data.to("cuda:0")

        res = stresslayer(data)
        resi = (res[0]-data.y).mean()+(res[1]-data.force).mean()+(res[2]-data.stress).mean()
        resi.backward()

    def test_force_direct_stressxyz(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=3, dim=16, cutoff=6.0, n_block=2, get_force=True,
                 direct_force=True, get_stress=True, readout="mean", mode="xyz", try_add_edge_msg=True,
                 )
        stresslayer.to("cuda:0")
        data.to("cuda:0")

        res = stresslayer(data)
        res = res[0].mean()+res[1].mean()+res[2].mean()
        res.backward()

    def test_force_direct_stressxyz2(self):

        data = self.data
        stresslayer = CGGRUForceStress(nfeat_node=19, nc_edge_hidden=3, dim=16, cutoff=6.0, n_block=2, get_force=True,
                 direct_force=True, get_stress=True, mode="xyz", try_add_edge_msg=True,
                 )
        stresslayer.to("cuda:0")
        data.to("cuda:0")

        res = stresslayer(data)
        resi = (res[0]-data.y).mean()+(res[1]-data.force).mean()+(res[2]-data.stress).mean()
        resi.backward()

if __name__ == '__main__':
    unittest.main()
