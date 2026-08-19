"""HDF write helpers, and the content hash a data node is named by."""

import hashlib
from typing import Union

import h5py
import numpy.typing as npt
import numpy as np


def _hash(array: npt.ArrayLike) -> str:
    """A simple numpy hash function."""
    array = np.asarray(array)
    return hashlib.sha1(array.tobytes()).hexdigest()


def _hdf_write(
    h5: Union[h5py.File, h5py.Group],
    data: npt.NDArray,
    name: str = "data",
    group: str = "/data",
) -> h5py.Dataset:
    try:
        node = h5.require_dataset(
            "%s/%s" % (group, name), data.shape, data.dtype, exact=True
        )
    except TypeError:
        del h5[group][name]
        node = h5.create_dataset(
            "%s/%s" % (group, name), data.shape, data.dtype, exact=True
        )

    node[:] = data
    return node
