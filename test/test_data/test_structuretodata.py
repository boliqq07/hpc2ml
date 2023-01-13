import unittest

import numpy as np
from pymatgen.core import Structure

from hpc2ml.data.structuretodata import PAddSAPymatgen, PAddXPymatgen, PAddXASE, \
    StructureToData, PAddXArray, PAddPBCEdgeXYZ, PAddPBCEdgeSole

data = Structure.from_file("test_data/POSCAR")
datas = [data] * 10


class MyTestCase(unittest.TestCase):
    def test_AddSAPymatgen(self):
        addsap = PAddSAPymatgen()
        res = addsap.convert(data, y=None, state_attr=None)
        print(res)

    def test_AddSAPymatgen2(self):
        sub_converters = PAddXPymatgen()
        addsap = PAddSAPymatgen(sub_converters=[sub_converters])
        res = addsap.convert(data, y=None, state_attr=None)
        print(res)

    def test_AddXASE(self):
        sub_converters1 = PAddSAPymatgen()
        sub_converters2 = PAddXPymatgen()
        addsap = StructureToData(sub_converters=[sub_converters1, sub_converters2], n_jobs=2)
        # res = addsap.transform(datas, y=None, )
        res = addsap.transform(datas, y=None, state_attr=[[3, 4, 5]] * 10)
        print(res)

    def test_AddXArray(self):
        array = np.random.random((120, 10))
        sub_converters1 = PAddXASE()
        sub_converters2 = PAddXPymatgen()
        sub_converters3 = PAddXArray(array)

        addsap = StructureToData(sub_converters=[sub_converters1, sub_converters2, sub_converters3],
                                 n_jobs=1)
        # res = addsap.transform(datas, y=None, )
        res = addsap.transform(datas, y=None)
        print(res)

    def test_AddPBCEdgeDistance(self):
        addsap = StructureToData(sub_converters=[PAddXPymatgen(), PAddSAPymatgen(), PAddPBCEdgeXYZ()], n_jobs=2)
        res = addsap.transform_and_save(datas)
        print(res)

    def test_add(self):
        addsap = PAddXPymatgen()+PAddSAPymatgen()+PAddPBCEdgeXYZ()
        assert len(addsap.sub_converters)==3
        assert addsap.contain_base == False

    def test_add2(self):
        addsap = StructureToData()+PAddSAPymatgen()+PAddPBCEdgeXYZ()
        assert len(addsap.sub_converters)==2
        assert addsap.contain_base == True

    def test_add3(self):
        addsap = PAddSAPymatgen()+PAddSAPymatgen(contain_base=True)+PAddPBCEdgeXYZ()
        assert addsap.contain_base == True
        assert len(addsap.sub_converters)==3

    def test_add4(self):
        addsap = PAddSAPymatgen(contain_base=True)+PAddPBCEdgeXYZ()
        assert addsap.contain_base==True
        assert len(addsap.sub_converters) == 2

    # def test_add45(self):
    #     addsap = PAddSAPymatgen(PAddPBCEdgeXYZ(PAddPBCEdgeSole()))

    def test_add45(self):
        addsap = PAddSAPymatgen()+PAddPBCEdgeXYZ(PAddPBCEdgeSole())


if __name__ == '__main__':
    unittest.main()
