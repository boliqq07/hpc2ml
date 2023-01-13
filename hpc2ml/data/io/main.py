import os
from typing import Union, Sequence

import path
from mgetool.tool import parallelize


def sparse_source_data(source_file: str = "vasprun.xml", source_path: Union[Sequence, str] = ".", fmt: str="vasprun",
                       n_jobs=4, **kwargs):
    """
    Sparse data by different function.

    Args:
        source_file: file name to sparse.
        source_path: (str,list), path or paths.
        fmt: str, load function named  "sparse_{fmt}" in from ``hpc2ml.data.io`` .
        n_jobs: int, the parallel number to load,
        **kwargs: dict, the parameter in "sparse_{fmt}".

    Returns:
        data_dict:dict, data
    """

    if isinstance(source_path, str):
        source_path = [source_path, ]

    source_file = [os.path.join(pathi, source_file) for pathi in source_path]
    from hpc2ml.data import io
    func = getattr(io, f"sparse_{fmt}")

    def func2(i):
        try:
            dicti = func(i, **kwargs)
        except:
            dicti = {}
            print(f"Error for : {i}")
        return dicti

    dct = {}
    res = parallelize(n_jobs=n_jobs, func=func2, iterable=source_file, tq=True, desc="sparse the source data")
    [dct.update(i) for i in res]

    # for i in source_file:
    #     try:
    #         dcti = func(i)
    #         dct.update(dcti)
    #     except:
    #         print(f"Error: {i}.")

    return dct


def find_leaf_path(root_pt):
    """
    Find the leaf path.

    Args:
        root_pt: pt: (str, path.Path, os.PathLike,pathlib.Path), path.

    Returns:
        paths: (list), list of sub leaf path.

    """
    if not isinstance(root_pt, path.Path):
        root_pt = path.Path(root_pt)

    sub_disk = list(root_pt.walkdirs())

    par_disk = [i.parent for i in sub_disk]
    par_disk = list(set(par_disk))

    res = [i for i in sub_disk if i not in par_disk]
    return res
