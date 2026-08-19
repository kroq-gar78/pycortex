"""``BrainData``: an array, and the space it lives in.

Step 5 of DATASET_REFACTOR.md. ``VolumeData`` and ``VertexData`` collapse into
this one concrete class; the volume/vertex distinction becomes
``braindata.space``. What is left here is what does not depend on the geometry:
the array, movie-ness, the content hash, the numpy operators, and ``copy``.

``copy`` is the clearest gain. It was written per subclass, each hand-listing the
arguments needed to rebuild its own geometry -- ``VolumeData.copy`` passed
``subject``, ``xfmname`` and ``mask``, and got the mask from a private attribute
to avoid re-resolving it. Here the space is carried over unchanged, so one
implementation serves every space and there is nothing to re-resolve.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from ._hdf import _hash, _hdf_write  # noqa: F401  (_hash re-exported; it lived here)
from ._space import Space


class BrainData:
    """An array of values, and the :class:`Space` that says where they are.

    Parameters
    ----------
    data : ndarray or str or None
        The values. What shapes are acceptable is the space's business; a str is
        read with nibabel. ``None`` is accepted only by spaces that can produce a
        default array for it.
    space : Space
        Where the values live. Resolved against ``data`` on the way in, so a flat
        volume ends up holding the space of the mask it actually matches.
    """

    def __init__(self, data: Union[npt.NDArray, str, None], space: Space) -> None:
        if isinstance(data, str):
            import nibabel

            nib = nibabel.load(data)
            data = nib.get_fdata().T

        #: Where these values live. Resolved, so it describes *this* array.
        self.space = space.resolve_for(data)
        self._data = self.space.validate(data)

    # ------------------------------------------------------------------
    # the array
    # ------------------------------------------------------------------
    @property
    def data(self) -> npt.NDArray:
        if isinstance(self._data, h5py.Dataset):
            return self._data[()]
        return self._data

    @data.setter
    def data(self, data: npt.NDArray) -> None:
        self._data = data

    @property
    def subject(self) -> str:
        return self.space.subject

    @property
    def movie(self) -> bool:
        """Whether the array carries a leading time axis."""
        return self.space.is_movie(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one frame of dense data in this space."""
        return self.space.shape

    @property
    def dense(self) -> npt.NDArray:
        """The array over the whole geometry, with a leading frame axis.

        Published by the views as ``.volume`` and ``.vertices``, which is where
        those names belong: they are what a space calls this, not what it is.
        """
        return self.space.to_dense(self.data)

    @property
    def name(self) -> str:
        """Content-addressed name. This is the HDF node name and browser key."""
        return "__%s" % _hash(self.data)[:16]

    def __hash__(self) -> int:
        return hash(_hash(self.data))

    def copy(self, data: npt.NDArray) -> "BrainData":
        """A new ``BrainData`` over ``data``, in this one's space.

        Cheap, and correct for every space without being written per space: the
        space is carried over rather than rebuilt from a hand-written argument
        list, so nothing re-resolves a mask or reloads a surface.
        """
        return BrainData(data, self.space)

    def exp(self) -> "BrainData":
        return self.copy(np.exp(self.data))

    def uniques(self, collapse: bool = False):
        yield self

    def __getitem__(self, idx: Any) -> "BrainData":
        """The frame at ``idx``. Movie data only."""
        if not self.movie:
            raise TypeError("Cannot index non-movie data")
        return self.copy(self.data[idx])

    # ------------------------------------------------------------------
    # numpy operators
    #
    # Written out rather than generated with `setattr` in a classmethod, which
    # made `vol + 1` unresolvable to any static tool and gave `__neg__`/`__abs__`
    # the same binary signature as the rest.
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> "BrainData":
        return self.copy(self.data + other)

    def __sub__(self, other: Any) -> "BrainData":
        return self.copy(self.data - other)

    def __mul__(self, other: Any) -> "BrainData":
        return self.copy(self.data * other)

    def __truediv__(self, other: Any) -> "BrainData":
        return self.copy(self.data / other)

    def __floordiv__(self, other: Any) -> "BrainData":
        return self.copy(self.data // other)

    def __pow__(self, other: Any) -> "BrainData":
        return self.copy(self.data**other)

    def __neg__(self) -> "BrainData":
        return self.copy(-self.data)

    def __abs__(self) -> "BrainData":
        return self.copy(abs(self.data))

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: Optional[str] = None
    ) -> h5py.Dataset:
        if name is None:
            name = self.name
        dgrp = h5.require_group("/data")

        if name in dgrp and "__%s" % _hash(dgrp[name][()])[:16] == name:
            # Same content under the same content-addressed name; nothing to do.
            return h5.get("/data/%s" % name)

        node = _hdf_write(h5, self.data, name=name)
        node.attrs["subject"] = self.subject
        self.space.write_hdf_attrs(h5, node)
        return node

    def save(self, filename: Union[str, h5py.Group], name: Optional[str] = None) -> None:
        """Save into the HDF file ``filename`` under ``name``."""
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
            if ext not in (".hdf", ".h5", ".hf5"):
                raise TypeError("Unknown file type")
            with h5py.File(filename, "a") as h5:
                self._write_hdf(h5, name=name)
        elif isinstance(filename, h5py.Group):
            self._write_hdf(filename, name=name)

    def to_json(self, simple: bool = False) -> dict[str, Any]:
        """How the browser should read this array.

        Only the ``simple=True`` half lives here: the other half describes a
        *view*, which is a different object now.
        """
        if not simple:
            return {}
        sdict: dict[str, Any] = dict(
            name=self.name,
            subject=self.subject,
            min=float(np.nan_to_num(self.data).min()),
            max=float(np.nan_to_num(self.data).max()),
        )
        sdict.update(self.space.describe_layout(self.data))
        return sdict

    def __repr__(self) -> str:
        return "<data in %r>" % (self.space,)


#: Kept importable, per the compatibility stance in DATASET_REFACTOR.md §4.1.
#:
#: These are now aliases rather than classes: the volume/vertex distinction is
#: ``braindata.space``, so there is nothing left for a subclass to carry. One
#: consequence is not preserved and cannot be: ``isinstance(vol, VolumeData)``
#: used to be true for a ``Volume``, and a view no longer *is* a ``BrainData`` --
#: it holds one. Ask ``isinstance(view.space, VolumeSpace)`` instead, which is
#: also the question that keeps working when a third space is added.
VolumeData = BrainData
VertexData = BrainData
