"""``Dataview``: how values become colors, composed with the values.

Steps 2, 3 and 5 of DATASET_REFACTOR.md. The three layers are

    Dataview          cmap, vmin/vmax, description, state, priority
       holds BrainData    the values
          holds Space     the geometry

related by composition only. A ``Dataview`` never inherits ``BrainData``, which
is what the old ``Volume(VolumeData, Dataview)`` did and what made
``BrainData.to_json`` and ``VolumeData.copy`` call ``super()`` methods that
existed nowhere in their own ancestry, resolving only because a subclass's MRO
happened to thread through the other base.

Two things follow immediately:

**One ``__init__`` for the view fields.** ``Dataview2D.__init__`` and
``DataviewRGB.__init__`` re-set ``state``/``attrs``/``priority``/``description``
by hand, neither calling ``Dataview.__init__``; the ``priority`` default of 1 was
written out three times. They call it now.

**No cooperative kwargs chain.** ``cmap``/``vmin``/``vmax`` used to travel as
opaque ``**kwargs`` through two classes with no idea what they were, landing in
``Dataview.attrs`` if unrecognized -- so ``Volume(data, subj, xfm, cmpa="hot")``
was a silently ignored attribute. The scalar column takes them by name.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, Self
else:
    from typing import NotRequired, Self

import h5py
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Colormap

from .. import options
from ._hdf import _hash
from ._space import MaskSpec, Space, SurfaceSpace, VolumeSpace, detect_space
from .braindata import BrainData

default_cmap = options.config.get("basic", "default_cmap")

# register_cmap is deprecated in matplotlib > 3.7.0 and replaced by colormaps.register
try:
    from matplotlib import colormaps as cm

    def register_cmap(cmap):
        return cm.register(cmap)

except ImportError:
    from matplotlib.cm import register_cmap


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


class ColormapDict(TypedDict):
    cmap: Colormap
    vmin: Optional[float]
    vmax: Optional[float]


class DataviewJSON(TypedDict, total=False):
    """The wire format consumed by ``webgl/resources/js/dataset.js``.

    The JS dispatches on the *shape* of these values -- ``mosaic`` absent means
    surface data, a nested list in ``data`` means a 2D view, ``raw`` true means
    4-channel uint8 -- so the shapes are load-bearing.
    """

    state: Any
    attrs: dict[str, Any]
    desc: str
    cmap: Optional[list[Any]]
    vmin: Optional[list[Any]]
    vmax: Optional[list[Any]]
    name: str
    subject: str
    min: float
    max: float
    raw: bool
    mosaic: tuple[int, int]
    shape: tuple[int, ...]
    data: list[Any]
    xfm: list[Any]
    split: int
    frames: int


def u(s, encoding: str = "utf8"):
    try:
        return s.decode(encoding)
    except AttributeError:
        return s


def _dumps(obj: Any) -> str:
    """``json.dumps`` that survives a numpy scalar.

    Of the numpy scalar types only ``np.float64`` and ``np.str_`` subclass a
    builtin json understands, so float32 data -- whose defaulted ``vmin`` is an
    ``np.float32`` -- could not be saved at all.
    """

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError("Object of type %s is not JSON serializable" % type(value))

    return json.dumps(obj, default=default)


class Dataview(ABC):
    """Abstract root: display metadata, and the ability to render to color.

    Deliberately holds no array and no colormap. An RGB view has no single
    colormap, and the previous design's answer to that was to skip
    ``Dataview.__init__`` entirely and let ``except AttributeError`` carry the
    control flow in ``to_json`` and ``_write_hdf``.
    """

    #: NaN positions captured before a scalar view was colormapped to uint8.
    #: uint8 cannot hold NaN, so this is the side channel that keeps
    #: "NaN implies transparent" working.
    _nan_mask: Optional[npt.NDArray[np.bool_]] = None

    def __init__(
        self,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        **attrs: Any,
    ) -> None:
        self.description = u(description)
        self.state = state
        self.attrs: dict[str, Any] = dict(attrs)
        self.attrs.setdefault("priority", priority)

    # ------------------------------------------------------------------
    # what every view has
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def space(self) -> Space:
        """Where this view's values live.

        The open axis: adding a kind of brain data means adding a
        :class:`~cortex.dataset._space.Space`, and consumers ask this rather than
        asking what class they are holding.
        """

    @property
    def subject(self) -> str:
        return self.space.subject

    @property
    @abstractmethod
    def raw(self) -> "Dataview":
        """This view rendered to 8-bit RGBA channels."""

    @abstractmethod
    def uniques(self, collapse: bool = False):
        """Yield the distinct data-carrying views inside this one."""

    @property
    def priority(self) -> Any:
        return self.attrs["priority"]

    @priority.setter
    def priority(self, value: Any) -> None:
        self.attrs["priority"] = value

    def get_cmapdict(self) -> Any:
        """Colormap arguments suitable for splatting into ``imshow``."""
        return {}

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        if simple:
            return DataviewJSON()
        return DataviewJSON(
            state=self.state, attrs=self.attrs.copy(), desc=self.description
        )

    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        """Slots 2-4. Overridden by the columns that carry a colormap."""
        view[2] = "null"
        view[3:5] = "null"

    def _write_view_node(
        self,
        h5: Union[h5py.File, h5py.Group],
        name: str,
        data: list[Any],
        xfmname: Optional[list[Any]],
    ) -> h5py.Dataset:
        """The eight-slot ``/views`` record.

        A hard interface: ``dataset.js`` reads it structurally, so do not reorder
        or retype. See the slot table in DATASET_REFACTOR.md's notes on the wire
        format.
        """
        views = h5.require_group("/views")
        view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
        view[0] = _dumps(data)
        view[1] = self.description
        self._write_cmap_slots(view)
        view[5] = _dumps(self.state)
        view[6] = _dumps(self.attrs)
        view[7] = _dumps(xfmname)
        return view

    @abstractmethod
    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        """Write both the data node(s) and the view node for this view."""

    def save(self, filename: Union[str, h5py.Group], name: str = "data") -> None:
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
            if ext not in (".hdf", ".h5", ".hf5"):
                raise TypeError("Unknown file type")
            with h5py.File(filename, "a") as h5:
                self._write_hdf(h5, name=name)
        elif isinstance(filename, h5py.Group):
            self._write_hdf(filename, name=name)

    @staticmethod
    def from_hdf(node: h5py.Dataset, subject: Optional[str] = None) -> "Dataview":
        data = json.loads(u(node[0]))
        # Decoded here rather than left as the bytes h5py hands back for a vlen
        # str: every one of the six classes used to reload with a `bytes`
        # description, since this went straight into the constructor.
        desc = u(node[1])
        try:
            cmap = json.loads(u(node[2]))
        except ValueError:
            cmap = u(node[2])
        vmin = json.loads(u(node[3]))
        vmax = json.loads(u(node[4]))
        state = json.loads(u(node[5]))
        attrs = json.loads(u(node[6]))
        try:
            xfmname = json.loads(u(node[7]))
        except ValueError:
            xfmname = None

        if not isinstance(vmin, list):
            vmin = [vmin]
        if not isinstance(vmax, list):
            vmax = [vmax]
        if not isinstance(cmap, list):
            cmap = [cmap]

        if len(data) != 1:
            # `Multiview` was never implemented -- its `__init__` raised on its
            # third line -- and this branch built a list of views and then raised
            # unconditionally. Both are deleted; this says so.
            raise NotImplementedError(
                "Views containing more than one dataview are not supported"
            )

        # An RGB view writes JSON null into slots 2-4 because it has no colormap.
        # Reading that back as `cmap=None` and passing it on meant the RGB
        # constructors, which have no `cmap` parameter, were handed one -- which
        # is why both factories below used to filter their kwargs down to a
        # hardcoded whitelist to get rid of it again.
        cmap_kwargs: dict[str, Any] = {}
        if cmap[0] is not None:
            cmap_kwargs = dict(cmap=cmap[0], vmin=vmin[0], vmax=vmax[0])

        return _from_hdf_view(
            node.file,
            data[0],
            xfmname=None if xfmname is None else xfmname[0],
            description=desc,
            state=state,
            subject=subject,
            **cmap_kwargs,
            **attrs,
        )


class DataviewScalar(Dataview):
    """One array of scalar values, displayed through a 1D colormap.

    Composes a :class:`BrainData` rather than being one. ``data``, ``dense`` and
    the numpy operators forward to it, which is the compatibility layer
    DATASET_REFACTOR.md §4.1 budgets for: ``.data`` is public and stays public.
    """

    def __init__(
        self,
        braindata: BrainData,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        **attrs: Any,
    ) -> None:
        self.braindata = braindata
        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(
            description=description, state=state, priority=priority, **attrs
        )
        # One rule, in one place. This used to be done eagerly here *and* lazily
        # in `Dataview.to_json` if still None *and* pulled from the channels by
        # `Dataview2D`, so whether `.vmin` was a number after construction
        # depended on which class built the object.
        self._resolve_percentiles()

    def _resolve_percentiles(self) -> None:
        """Default unset bounds to the 1st/99th data percentiles.

        Keeps ``np.percentile``'s numpy scalar rather than converting to a Python
        float: under NEP 50 a numpy scalar is a *strong* operand, so
        ``float32_channel -= vmin`` computes in float64 and rounds once where a
        weak Python float computes in float32. That difference is a single LSB on
        a few voxels -- but node names are content hashes, so it would silently
        change on-disk identity.
        """
        if self.vmin is None:
            self.vmin = np.percentile(np.nan_to_num(self.data), 1)
        if self.vmax is None:
            self.vmax = np.percentile(np.nan_to_num(self.data), 99)

    # ------------------------------------------------------------------
    # forwarded to the BrainData
    # ------------------------------------------------------------------
    @property
    def space(self) -> Space:
        return self.braindata.space

    @property
    def data(self) -> npt.NDArray:
        return self.braindata.data

    @data.setter
    def data(self, value: npt.NDArray) -> None:
        self.braindata.data = value

    @property
    def movie(self) -> bool:
        return self.braindata.movie

    @property
    def shape(self) -> tuple[int, ...]:
        return self.braindata.shape

    @property
    def dense(self) -> npt.NDArray:
        """The array a renderer samples, with a leading frame axis."""
        return self.braindata.dense

    @property
    def name(self) -> str:
        return self.braindata.name

    def __hash__(self) -> int:
        return hash(self.braindata)

    def uniques(self, collapse: bool = False):
        yield self

    def copy(self, data: Optional[npt.NDArray] = None) -> Self:
        """A view of the same kind over ``data``, keeping this one's space.

        The space is carried through :meth:`BrainData.copy`, so this needs no
        per-space argument list -- which is what ``VolumeData.copy`` and
        ``VertexData.copy`` each were.
        """
        if data is None:
            data = self.data
        return type(self)._from_parts(
            self.braindata.copy(data),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            description=self.description,
            state=self.state,
            **self.attrs,
        )

    @classmethod
    def _from_parts(cls, braindata: BrainData, **kwargs: Any) -> Self:
        """Build without going through the public constructor's signature.

        The public constructors take a space's arguments positionally --
        ``Volume(arr, "S1", "fullhead")`` -- so anything space-agnostic has to
        come in this way instead.
        """
        view = cls.__new__(cls)
        DataviewScalar.__init__(view, braindata, **kwargs)
        return view

    def exp(self) -> Self:
        return self.copy(np.exp(self.data))

    def __getitem__(self, idx: Any) -> Self:
        return self.copy(self.data[idx])

    def __add__(self, other: Any) -> Self:
        return self.copy(self.data + other)

    def __sub__(self, other: Any) -> Self:
        return self.copy(self.data - other)

    def __mul__(self, other: Any) -> Self:
        return self.copy(self.data * other)

    def __truediv__(self, other: Any) -> Self:
        return self.copy(self.data / other)

    def __floordiv__(self, other: Any) -> Self:
        return self.copy(self.data // other)

    def __pow__(self, other: Any) -> Self:
        return self.copy(self.data**other)

    def __neg__(self) -> Self:
        return self.copy(-self.data)

    def __abs__(self) -> Self:
        return self.copy(abs(self.data))

    # ------------------------------------------------------------------
    # color
    # ------------------------------------------------------------------
    def get_cmapdict(self) -> ColormapDict:
        return ColormapDict(cmap=_lookup_cmap(self.cmap), vmin=self.vmin, vmax=self.vmax)

    def _colormap_to_rgba(
        self,
    ) -> tuple[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray[np.bool_]]:
        from matplotlib import cm, colors

        cmap = self.get_cmapdict()["cmap"]
        norm = colors.Normalize(self.vmin, self.vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        # Captured before the uint8 conversion, which cannot represent NaN.
        nan_mask: npt.NDArray[np.bool_] = np.isnan(self.data)
        colored = cmapper.to_rgba(self.data.flatten()).reshape(self.data.shape + (4,))
        colored = (np.clip(colored, 0, 1) * 255).astype(np.uint8)
        colored[nan_mask, 3] = 0
        r, g, b, a = np.rollaxis(colored, -1)
        return (r, g, b, a), nan_mask

    @property
    def raw(self) -> "Dataview":
        """This view colormapped, as an RGB view of the same space.

        Names no concrete RGB class: the space knows which one views its data, so
        one implementation serves every space. ``Volume.raw`` and ``Vertex.raw``
        were identical but for that name.
        """
        (r, g, b, a), nan_mask = self._colormap_to_rgba()
        space = self.space
        wrap = type(space).scalar_view._from_parts
        # Fresh views rather than `self.copy`, which would carry this view's
        # `vmin`/`vmax` onto channels that are already uint8 colors. Harmless for
        # the three colors, which pass through unscaled -- but the alpha channel is
        # masked by assigning `alpha.vmin` into it, and writing a negative vmin
        # into a uint8 array raises. `alpha` stays an array so the RGB column wraps
        # it with the 0..1 bounds that assignment needs.
        result = type(space).rgb_view(
            wrap(BrainData(r, space)),
            wrap(BrainData(g, space)),
            wrap(BrainData(b, space)),
            alpha=a,
            description=self.description,
            state=self.state,
            priority=self.priority,
        )
        result._nan_mask = nan_mask
        return result

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        sdict = super().to_json(simple=simple)
        if simple:
            sdict.update(cast(DataviewJSON, self.braindata.to_json(simple=True)))
            return sdict
        # No percentile fallback: `__init__` resolved these, so a None here would
        # mean something set them back afterwards and substituting would hide it.
        sdict.update(
            DataviewJSON(
                cmap=[self.cmap], vmin=[self.vmin], vmax=[self.vmax], data=[self.name]
            )
        )
        sdict.update(cast(DataviewJSON, self.space.to_json()))
        return sdict

    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        view[2] = _dumps([self.cmap])
        view[3] = _dumps([self.vmin])
        view[4] = _dumps([self.vmax])

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self.braindata._write_hdf(h5)
        return self._write_view_node(
            h5, name, [self.name], self.space.view_xfmname
        )

    @classmethod
    def empty(cls, subject: str, *spec: Any, value: float = 0, **kwargs: Any) -> Self:
        raise NotImplementedError

    @staticmethod
    def _sample(shape: tuple[int, ...], value: Optional[float]) -> npt.NDArray:
        """A fresh array: constant ``value``, or standard normal if None."""
        if value is None:
            return np.random.randn(*shape)
        return np.ones(shape) * value


#: The channel type of a composite view. One TypeVar, covariant, bound to the
#: scalar column -- so ``Volume2D.dim1`` is a ``Volume`` and ``VertexRGB.alpha`` is
#: a ``Vertex`` without either class re-declaring anything.
#:
#: Covariance is what lets ``Dataview2D[DataviewScalar]`` accept a ``Volume2D``,
#: which an invariant parameter would reject. It is sound because the channels are
#: read-only properties backed by private fields, set once in ``__init__``.
#:
#: This is the *cheap* half of typing the composite columns. It cannot narrow
#: ``.raw``: that would need the space to be generic over its whole view family and
#: threaded through every column, because a type parameter cannot be projected out
#: of another type parameter. Each concrete class restates ``raw``'s return type
#: instead -- see COMPOSITION.md.
ScalarT = TypeVar("ScalarT", bound="DataviewScalar", covariant=True)


def _lookup_cmap(name: Any) -> Colormap:
    """A matplotlib colormap by matplotlib name, pycortex name, or instance."""
    from matplotlib import colors
    from matplotlib import pyplot as plt

    try:
        return plt.get_cmap(name)
    except ValueError:
        cmapdir = options.config.get("webgl", "colormaps")
        colormaps = {
            os.path.split(c)[1][:-4]: c for c in glob.glob(os.path.join(cmapdir, "*.png"))
        }
        if name not in colormaps:
            raise ValueError("Unknown color map %s" % name)
        image = plt.imread(colormaps[name])
        cmap = colors.ListedColormap(
            np.squeeze(image), name=name if isinstance(name, str) else name.name
        )
        register_cmap(cmap)
        return cmap


class Volume(DataviewScalar):
    """
    Encapsulates a 3D volume or 4D volumetric movie. Includes information on how
    the volume should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 3D with shape (z,y,x), 1D with shape (v,) for masked data,
        4D with shape (t,z,y,x), or 2D with shape (t,v). For masked data, if the
        size of the given array matches any of the existing masks in the database,
        that mask will automatically be loaded. If it does not, an error will be
        raised.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    mask : ndarray, optional
        Binary 3D array with shape (z,y,x) showing which voxels are selected.
        If masked data is given, the mask will automatically be loaded if it
        exists in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        xfmname: Union[str, bytes],
        mask: MaskSpec = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        # The whole job: build the space, build the data, hand them up.
        super().__init__(
            BrainData(data, VolumeSpace(subject, xfmname, mask)),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # the vocabulary a volumetric space exposes, one line each
    # ------------------------------------------------------------------
    @property
    def space(self) -> VolumeSpace:
        return cast(VolumeSpace, self.braindata.space)

    @property
    def xfmname(self) -> str:
        return self.space.xfmname

    @property
    def mask(self) -> Optional[npt.NDArray[np.bool_]]:
        return self.space.mask

    @property
    def linear(self) -> bool:
        return self.space.linear

    @property
    def volume(self) -> npt.NDArray:
        """3D or 4D volume, automatically unmasking masked data."""
        return self.dense

    @property
    def masked(self) -> "_masker":
        return _masker(self)

    @classmethod
    def empty(cls, subject: str, xfmname: str, value: float = 0, **kwargs: Any) -> "Volume":
        shape = VolumeSpace(subject, xfmname).template_shape
        return cls(cls._sample(shape, value), subject, xfmname, **kwargs)

    @classmethod
    def random(cls, subject: str, xfmname: str, **kwargs: Any) -> "Volume":
        shape = VolumeSpace(subject, xfmname).template_shape
        return cls(cls._sample(shape, None), subject, xfmname, **kwargs)

    def map(self, projection: str = "nearest") -> "Vertex":
        """This volume projected onto the surface."""
        from cortex import utils

        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        result = mapper(self)
        result.vmin = self.vmin
        result.vmax = self.vmax
        result.cmap = self.cmap
        return result

    def save_nii(self, filename: os.PathLike) -> None:
        """Save as a nifti file, with headers from the transform's reference."""
        from ..database import db

        import nibabel

        affine = db.get_xfm(self.subject, self.xfmname).reference.affine
        nibabel.save(nibabel.Nifti1Image(self.volume.T, affine), filename)

    def __repr__(self) -> str:
        space = self.space
        maskstr = "volumetric"
        if space.linear:
            name = space.mask_name
            if name is None:
                name = "custom"
            maskstr = "%s masked" % name
        if self.movie:
            maskstr += " movie"
        return "<%s data for (%s, %s)>" % (
            maskstr[0].upper() + maskstr[1:],
            self.subject,
            self.xfmname,
        )


class Vertex(DataviewScalar):
    """
    Encapsulates a 1D vertex map or 2D vertex movie. Includes information on how
    the data should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 1D with shape (v,), or 2D with shape (t,v). Here, v can
        be the number of vertices in both hemispheres, or the number of vertices
        in either one of the hemispheres. In that case, the data for the other
        hemisphere will be filled with zeros.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            BrainData(data, SurfaceSpace(subject)),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )

    @property
    def space(self) -> SurfaceSpace:
        return cast(SurfaceSpace, self.braindata.space)

    @property
    def llen(self) -> int:
        return self.space.llen

    @property
    def rlen(self) -> int:
        return self.space.rlen

    @property
    def nverts(self) -> int:
        return self.space.nverts

    @property
    def vertices(self) -> npt.NDArray:
        """Per-vertex values, with a leading frame axis."""
        return self.dense

    @property
    def left(self) -> npt.NDArray:
        return self.space.split_hemispheres(self.data)[0]

    @property
    def right(self) -> npt.NDArray:
        return self.space.split_hemispheres(self.data)[1]

    @classmethod
    def empty(cls, subject: str, value: float = 0, **kwargs: Any) -> "Vertex":
        shape = SurfaceSpace(subject).template_shape
        return cls(cls._sample(shape, value), subject, **kwargs)

    @classmethod
    def random(cls, subject: str, **kwargs: Any) -> "Vertex":
        shape = SurfaceSpace(subject).template_shape
        return cls(cls._sample(shape, None), subject, **kwargs)

    def volume(self, xfmname: str, projection: str = "nearest", **kwargs: Any) -> Volume:
        """This data mapped back into a volume space. Not particularly accurate."""
        import warnings

        from cortex import utils

        warnings.warn("Inverse mapping cannot be accurate")
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def map(
        self,
        target_subj: str,
        surface_type: str = "fiducial",
        hemi: Literal["lh", "rh", "both"] = "both",
        fs_subj: Optional[str] = None,
        **kwargs: Any,
    ) -> "Vertex":
        """Map this data from this surface to another subject's surface.

        NOTE: Requires either previous computation of mapping matrices
        (with ``cortex.db.get_mri_surf2surf_matrix``) or an active freesurfer
        environment.
        """
        if hemi not in ["lh", "rh", "both"]:
            raise ValueError("`hemi` kwarg must be 'lh', 'rh', or 'both'")
        from ..database import db

        mats = db.get_mri_surf2surf_matrix(
            self.subject,
            surface_type,
            hemi="both",
            target_subj=target_subj,
            fs_subj=fs_subj,
            **kwargs,
        )
        new_data = [mats[0].dot(self.left), mats[1].dot(self.right)]
        if hemi == "both":
            stacked = np.hstack(new_data)
        elif hemi == "lh":
            stacked = np.hstack([new_data[0], np.nan * np.zeros(new_data[1].shape)])
        else:
            stacked = np.hstack([np.nan * np.zeros(new_data[0].shape), new_data[1]])
        return Vertex(
            stacked, target_subj, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
        )

    def __repr__(self) -> str:
        return "<Vertex %sdata for %s>" % ("movie " if self.movie else "", self.subject)


class _masker:
    def __init__(self, dv: Volume) -> None:
        self.dv = dv
        self.data = dv.data if dv.linear else None

    def __getitem__(self, masktype: str) -> Volume:
        from ..database import db

        mask = db.get_mask(self.dv.subject, self.dv.xfmname, masktype)
        return self.dv.copy(self.dv.volume[:, mask].squeeze())


# ----------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------
@overload
def normalize(data: tuple[Any, Any, Any]) -> Union[Volume, "VolumeRGB"]: ...


@overload
def normalize(data: tuple[Any, Any]) -> Vertex: ...


@overload
def normalize(data: Dataview) -> Dataview: ...


def normalize(data: Union[Dataview, tuple]) -> Dataview:
    if isinstance(data, Dataview):
        return data
    if isinstance(data, tuple):
        if len(data) == 3:
            if data[0].dtype == np.uint8:
                return VolumeRGB(
                    data[0][..., 0], data[0][..., 1], data[0][..., 2], *data[1:]
                )
            return Volume(*data)
        if len(data) == 2:
            return Vertex(*data)
    raise TypeError("Invalid input for Dataview")


def _from_hdf_data(
    h5: h5py.File,
    name: str,
    xfmname: Optional[str] = None,
    subject: Optional[str] = None,
    **kwargs: Any,
) -> Dataview:
    """Decode a ``__hash``-named data node into the view of its space."""
    dnode = h5.get("/data/%s" % name)
    if dnode is None:
        dnode = h5.get(name)

    attrs = {k: u(v) for (k, v) in dnode.attrs.items()}
    if subject is None:
        subject = attrs["subject"]
    # support old style xfmname saving as attribute
    if xfmname is None and "xfmname" in attrs:
        xfmname = attrs["xfmname"]
    mask: MaskSpec = None
    if "mask" in attrs:
        if attrs["mask"].startswith("__"):
            mask = h5[
                "/subjects/%s/transforms/%s/masks/%s"
                % (attrs["subject"], xfmname, attrs["mask"])
            ][()]
        else:
            mask = attrs["mask"]

    space = detect_space(attrs, subject=subject, xfmname=xfmname, mask=mask)

    # support old style RGB volumes: uint8 with a trailing channel axis
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        channels = [_wrap(space, dnode[..., i]) for i in range(3)]
        alpha = _wrap(space, dnode[..., 3]) if dnode.shape[-1] == 4 else None
        # The *view* record said "scalar", so it carried a colormap and bounds,
        # and only opening the node revealed packed RGB. Those three arguments
        # describe a colormap this view does not have, so they are meaningless
        # here rather than merely unaccepted -- the one kwarg drop that is not a
        # workaround for the construction path.
        rgb_kwargs = {k: v for k, v in kwargs.items() if k not in ("cmap", "vmin", "vmax")}
        return type(space).rgb_view(*channels, alpha=alpha, **rgb_kwargs)

    return _wrap(space, dnode, **kwargs)


def _wrap(space: Space, data: Any, **kwargs: Any) -> DataviewScalar:
    """A scalar view over ``data`` in ``space``, naming no concrete class."""
    return type(space).scalar_view._from_parts(BrainData(data, space), **kwargs)


def _from_hdf_view(
    h5: h5py.File,
    data: Any,
    xfmname: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    subject: Optional[str] = None,
    **kwargs: Any,
) -> Dataview:
    if isinstance(data, str):
        return _from_hdf_data(
            h5, data, xfmname=xfmname, vmin=vmin, vmax=vmax, subject=subject, **kwargs
        )

    # A surface view has no transform, so slot 7 is null and `xfmname` arrives as
    # None rather than as a per-channel list. Indexing it unconditionally raised
    # TypeError here, and `Dataset.from_file` swallows per-view exceptions -- so a
    # saved Vertex2D was silently *dropped* on reload.
    xfmnames = xfmname if isinstance(xfmname, (list, tuple)) else [xfmname] * len(data)

    channels = [
        _from_hdf_data(
            h5, node, xfmname=xfmnames[i if i < len(xfmnames) else 0], subject=subject
        )
        if node is not None
        else None
        for i, node in enumerate(data)
    ]
    first = channels[0]
    assert first is not None
    space = first.space

    if len(data) == 2:
        return type(space).twod_view(
            channels[0],
            channels[1],
            vmin=vmin[0],
            vmin2=vmin[1],
            vmax=vmax[0],
            vmax2=vmax[1],
            **kwargs,
        )
    if len(data) == 4:
        return type(space).rgb_view(
            channels[0], channels[1], channels[2], alpha=channels[3], **kwargs
        )
    raise ValueError("Invalid Dataview specification")


from .view2D import Dataview2D, Vertex2D, Volume2D  # noqa: E402
from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB  # noqa: E402

# The space is what a space-agnostic method dispatches through, so each space
# names the three classes that view its data. Assigned here, after all three
# columns exist, rather than inside the classes: it is the one edge in the graph
# that genuinely points both ways.
VolumeSpace.scalar_view = Volume
VolumeSpace.twod_view = Volume2D
VolumeSpace.rgb_view = VolumeRGB
SurfaceSpace.scalar_view = Vertex
SurfaceSpace.twod_view = Vertex2D
SurfaceSpace.rgb_view = VertexRGB
