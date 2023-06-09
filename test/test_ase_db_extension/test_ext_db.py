import os
import pathlib
import sqlite3
import unittest

from ase.db import connect

from hpc2ml.db.ase_db_extension.ext_db import db_cat, db_rename, db_transform


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.db = connect("test_ase_db_extension/organometal.db")

    def test_data_cat(self):

        if os.path.isfile("test_ase_db_extension/new2.db"):
            os.remove("test_ase_db_extension/new2.db")

        db2 = self.db
        db1 = connect("test_ase_db_extension/temp.db")
        try:
            db3 = db_cat(db1, db2, new_database_name="test_ase_db_extension/new2.db")
        except sqlite3.IntegrityError:
            pass

    def test_rename(self):

        ar = db_rename(self.db, name_pair=(("space_group2", "space_group"),), check=True)
        print(ar[1].key_value_pairs)
        print(ar[1].data)

        ar = db_rename(self.db, name_pair=(("space_group2", "space_group1"),), check=True)

    def test_transform(self):

        if os.path.isfile("data.json"):
            os.remove("data.json")

        db = self.db
        new_base = db_transform(db, "data.json")

        new_base = connect("data.json")
        print(new_base[1].toatoms())
        print(new_base[1])


if __name__ == '__main__':
    unittest.main()
