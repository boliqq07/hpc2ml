import unittest

from hpc2ml.data.structuretodata import PAddSAPymatgen, PAddXPymatgen, StructureToData, PAddPBCEdgeXYZ, PAddForce


class MyTestCase(unittest.TestCase):

    def test_from_vasp(self):
        addsap = StructureToData(sub_converters=[PAddXPymatgen(), PAddSAPymatgen(), PAddPBCEdgeXYZ(), PAddForce()],
                                 n_jobs=4)
        source_path = ["test_data/pure_opt", "test_data/OH_add_static"]
        addsap.sparse_source_data(source_file="vasprun.xml", source_path=source_path, fmt="vasprun")
        res = addsap.transform_data_dict_and_save()
        print(res)


if __name__ == '__main__':
    unittest.main()
