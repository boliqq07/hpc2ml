import unittest

from hpc2ml.data.structuretodata import PAddSAPymatgen, PAddXPymatgen, StructureToData, PAddPBCEdgeXYZ, PAddXEmbeddingDict


class MyTestCase(unittest.TestCase):

    def test_from_ase_db(self):
        addsap = StructureToData(sub_converters=[PAddXPymatgen(), PAddSAPymatgen(), PAddPBCEdgeXYZ(), PAddXEmbeddingDict()],
                                 n_jobs=1)

        addsap.sparse_source_data(source_file="test_ase_db_extension/organometal.db", source_path=".", fmt="ase")
        res = addsap.transform_data_dict_and_save()
        print(res)

    def test_from_ase_json(self):
        addsap = StructureToData(sub_converters=[PAddXPymatgen(), PAddSAPymatgen(), PAddPBCEdgeXYZ(), PAddXEmbeddingDict()],
                                 n_jobs=1)
        addsap.sparse_source_data(source_file="test_data/data.json", source_path=".", fmt="ase")
        res = addsap.transform_data_dict_and_save()
        print(res)


if __name__ == '__main__':
    unittest.main()
