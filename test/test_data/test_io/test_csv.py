import unittest

from hpc2ml.data.structuretodata import PAddSAPymatgen, PAddXPymatgen, StructureToData, PAddPBCEdgeXYZ, PAddXEmbeddingDict


class MyTestCase(unittest.TestCase):

    def test_from_csv(self):
        addsap = StructureToData(sub_converters=[PAddXPymatgen(), PAddSAPymatgen(), PAddPBCEdgeXYZ(), PAddXEmbeddingDict()],
                                 n_jobs=4)
        addsap.sparse_source_data(source_file="test_data/kim_raw_data.csv", source_path=".", fmt="csv")
        res = addsap.transform_data_dict_and_save()
        print(res)


if __name__ == '__main__':
    unittest.main()
