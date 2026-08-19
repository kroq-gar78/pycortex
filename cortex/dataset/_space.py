"""``Space``: where a value lives, with no values in it.

Step 1 of DATASET_REFACTOR.md. The geometry that ``VolumeData`` and
``VertexData`` each derived for themselves -- a shape from ``db.get_xfm``, vertex
counts from ``db.get_surf``, a mask guessed from a voxel count -- moves here, so
that adding a new kind of brain data means adding a space rather than
reimplementing colormapping, HDF and JSON three more times.

Three properties, in the order they matter:

**Spaces are values.** Immutable, hashable, comparable. ``space1 == space2``
replaces the ad-hoc subject/xfmname/mask-equality checks that were spelled out
by hand in ``Volume2D.raw``, ``VolumeRGB.__init__`` and ``Vertex2D.__init__``,
each slightly differently. Being a value is also what makes the next property
safe.

**This is the only module that touches ``db``.** ``VertexData.__init__`` called
``db.get_surf`` for *every* ``Vertex`` constructed, just to learn two integers
that depend on nothing but the subject. Those lookups are cached here, keyed on
what they actually depend on.

**Geometry that depends on the array is resolved, not stored.** A flat array's
mask and a half-length vertex array's hemisphere are facts about *that array*, so
they cannot live on a shared value. ``resolve_for(data)`` returns the space that
array is really in -- a new value, never a mutation -- and :class:`BrainData`
holds the result. That is the one place the old ``_check_size`` mixed
data-derived facts (``linear``, ``movie``) with space-derived ones (``shape``,
``mask``), and it is the only part of this extraction that is not a pure move.
"""

from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from ..database import db

if TYPE_CHECKING:
    from ._webgl import WebGLPayload
    from .braindata import BrainData

MaskSpec = Union[npt.NDArray, str, None]


@lru_cache(maxsize=None)
def _xfm_shape(subject: str, xfmname: str) -> tuple[int, ...]:
    """The reference shape of a transform. Cached: it is a database fact."""
    return tuple(db.get_xfm(subject, xfmname).shape)


@lru_cache(maxsize=None)
def _hemisphere_lengths(subject: str) -> tuple[int, int]:
    """Vertex counts per hemisphere. Cached; this ran once per ``Vertex`` built."""
    try:
        left, right = db.get_surf(subject, "wm")
    except IOError:
        left, right = db.get_surf(subject, "fiducial")
    return len(left[0]), len(right[0])


@lru_cache(maxsize=None)
def _named_mask(subject: str, xfmname: str, masktype: str) -> npt.NDArray[np.bool_]:
    return db.get_mask(subject, xfmname, masktype)


def _find_mask(nvox: int, subject: str, xfmname: str) -> tuple[str, npt.NDArray[np.bool_]]:
    """The database mask with exactly ``nvox`` voxels selected."""
    import nibabel

    files = db.get_paths(subject)["masks"].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_fdata().T != 0
        if nvox == np.sum(mask):
            found = re.compile(r"mask_(.+).nii.gz").search(os.path.split(fname)[1])
            assert found is not None, "Mask filename %s is not mask_<type>.nii.gz" % fname
            return found.group(1), mask

    raise ValueError("Cannot find a valid mask")


def _decode(value: Union[str, bytes]) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class Space(ABC):
    """Where a value lives: a subject, plus whatever else locates one value.

    Subclasses must be **immutable after construction**. Everything else here
    relies on it: equality, hashing, and the fact that a space can be shared
    between views instead of rebuilt per view.
    """

    #: Constructor arguments besides ``subject``, in positional order. Read by
    #: the composite views, which have to build a space when handed raw arrays,
    #: and by the error messages that then say which argument was missing.
    spec_keys: ClassVar[tuple[str, ...]] = ()

    #: Set by ``register_space``; a space registered as a fallback is consulted
    #: only after every other, which is how a file carrying no discriminator is
    #: still recognised.
    fallback: ClassVar[bool] = False

    def __init__(self, subject: Union[str, bytes]) -> None:
        self._subject = _decode(subject)

    @property
    def subject(self) -> str:
        return self._subject

    # ------------------------------------------------------------------
    # value semantics
    # ------------------------------------------------------------------
    @abstractmethod
    def _identity(self) -> tuple:
        """Everything that makes two spaces the same space."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return type(self) is type(other) and self._identity() == other._identity()

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash((type(self).__name__,) + self._identity())

    # ------------------------------------------------------------------
    # the classes that view data in this space
    #
    # A view never names a concrete class of another column, so `.raw` and the
    # HDF factories work for any space. These are filled in at the bottom of the
    # modules that define them, to keep the import graph acyclic.
    # ------------------------------------------------------------------
    scalar_view: ClassVar[Any] = None
    twod_view: ClassVar[Any] = None
    rgb_view: ClassVar[Any] = None

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def xfmname(self) -> Optional[str]:
        """The transform to sample this data through, or None.

        The one geometry fact every renderer asks for, which is why it is on the
        base rather than only on the volumetric space: consumers used to reach for
        it with ``hasattr(braindata, "xfmname")``.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of one frame of *dense* data in this space."""

    @property
    def template_shape(self) -> tuple[int, ...]:
        """Shape a *fresh* array should have. What ``empty``/``random`` read."""
        return self.shape

    @abstractmethod
    def resolve_for(self, data: Optional[npt.NDArray]) -> "Space":
        """The space ``data`` is really in.

        Returns ``self`` when nothing is left to resolve. Must not mutate: a
        space is a value, so learning which mask a flat array matches produces a
        *new* space.
        """

    @abstractmethod
    def validate(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Check ``data`` against this geometry and return it in stored form.

        May pad -- ``SurfaceSpace`` fills the absent hemisphere with zeros -- but
        must not otherwise transform. Call only on a resolved space.
        """

    @abstractmethod
    def is_movie(self, data: npt.NDArray) -> bool:
        """Whether ``data`` carries a leading time axis."""

    @abstractmethod
    def to_dense(self, data: npt.NDArray) -> npt.NDArray:
        """``data`` as one array per frame over the whole geometry.

        Published as ``.volume`` and ``.vertices``; unmasks a flat volume and
        adds the leading frame axis a single frame does not carry.
        """

    def align(self, first: "BrainData", second: "BrainData") -> tuple[npt.NDArray, npt.NDArray]:
        """Two arrays in a layout where position *i* means the same place in both.

        What a 2D view needs before it can colormap its two dimensions jointly.
        The stored arrays already serve any space in which one position is one
        location, which is why this is concrete.
        """
        return first.data, second.data

    # ------------------------------------------------------------------
    # rendering hooks -- so consumers stop branching on the space
    # ------------------------------------------------------------------
    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> "WebGLPayload":
        """``data`` encoded for the browser.

        Only two encodings exist, because ``dataset.js`` selects between them by
        testing ``mosaic === undefined``. Raising by default is deliberate: a
        space with no browser representation is a legitimate thing to have, since
        ``quickflat`` needs only :meth:`to_dense`.
        """
        raise TypeError(
            "%s has no webgl wire encoding, so data in it cannot be sent to a "
            "browser. Return one of the two encodings dataset.js can read from "
            "pack_for_webgl -- MosaicTexture or VertexAttributes -- or add a "
            "third to webgl/resources/js/dataset.js. quickflat can still draw it."
            % type(self).__name__
        )

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Space-specific keys for a view's full JSON description."""

    def describe_layout(self, data: npt.NDArray) -> dict[str, Any]:
        """Keys telling the browser how to unpack the array it is sent.

        Distinct from :meth:`to_json`, which describes the space rather than a
        particular array in it -- which is why only this one takes the data.
        Read by ``dataset.js``, so the keys are a hard interface.
        """
        return {}

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    @abstractmethod
    def write_hdf_attrs(self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset) -> None:
        """Record whatever :meth:`from_hdf` will need to rebuild this space."""

    @property
    def view_xfmname(self) -> Optional[list[str]]:
        """Slot 7 of the view record: ``[xfmname]``, or None for a space with none."""
        return None if self.xfmname is None else [self.xfmname]

    @classmethod
    @abstractmethod
    def from_hdf(
        cls, attrs: dict[str, Any], *, subject: str, xfmname: Optional[str], mask: MaskSpec
    ) -> Optional["Space"]:
        """This space rebuilt from a data node's attrs, or None if it is not ours."""

    # ------------------------------------------------------------------
    # construction from a partial spec, for the composite views
    # ------------------------------------------------------------------
    @classmethod
    def from_spec(cls, subject: Optional[str], **spec: Any) -> "Space":
        """Build from ``subject`` plus :attr:`spec_keys`, requiring each.

        A composite view handed raw arrays has to build the space itself, and
        "which arguments are mandatory" is the space's business rather than
        something each of four constructors states for itself.
        """
        if subject is None:
            raise TypeError("subject must be specified with raw data")
        missing = [k for k in cls.spec_keys if spec.get(k) is None]
        if missing:
            raise TypeError(
                "%s must be specified with raw data" % " and ".join(missing)
            )
        return cls(subject, *(spec[k] for k in cls.spec_keys))


_SPACES: list[type[Space]] = []


def register_space(space: type[Space]) -> type[Space]:
    """Register ``space`` for HDF detection, ahead of every fallback space.

    Order matters: detection takes the first space whose ``from_hdf`` returns
    non-None. Inserting ahead of the fallbacks means a space registered by a
    third party -- necessarily after this package has registered its own -- is
    still consulted before the catch-all that would otherwise claim its nodes.
    """
    if space.fallback:
        _SPACES.append(space)
    else:
        first_fallback = next(
            (i for i, s in enumerate(_SPACES) if s.fallback), len(_SPACES)
        )
        _SPACES.insert(first_fallback, space)
    return space


def registered_spaces() -> list[type[Space]]:
    return list(_SPACES)


def detect_space(
    attrs: dict[str, Any], *, subject: str, xfmname: Optional[str], mask: MaskSpec
) -> Space:
    """The space a data node is in, by asking each registered space in turn."""
    for cls in registered_spaces():
        space = cls.from_hdf(attrs, subject=subject, xfmname=xfmname, mask=mask)
        if space is not None:
            return space
    raise ValueError("No registered space claims this data node")


@register_space
class VolumeSpace(Space):
    """A voxel grid: subject, transform, and optionally a mask over it.

    ``mask`` is the *resolved* boolean array or None for a dense grid. What the
    user passed -- a name, an array, or nothing at all -- is kept as
    :attr:`mask_spec` because that is what gets written to HDF.
    """

    spec_keys = ("xfmname",)

    def __init__(
        self,
        subject: Union[str, bytes],
        xfmname: Union[str, bytes],
        mask: MaskSpec = None,
        _resolved: Optional[npt.NDArray[np.bool_]] = None,
        _mask_name: Optional[str] = None,
    ) -> None:
        super().__init__(subject)
        self._xfmname = _decode(xfmname)
        self._mask_spec = mask
        self._mask = _resolved
        self._mask_name = _mask_name

    def _identity(self) -> tuple:
        mask = self._mask
        return (
            self._subject,
            self._xfmname,
            None if mask is None else mask.tobytes(),
        )

    @property
    def xfmname(self) -> str:
        return self._xfmname

    @property
    def mask_spec(self) -> MaskSpec:
        """What was passed as ``mask``, which is what HDF records."""
        return self._mask_spec

    @property
    def mask(self) -> Optional[npt.NDArray[np.bool_]]:
        """The resolved boolean mask, or None for a dense grid."""
        return self._mask

    @property
    def mask_name(self) -> Optional[str]:
        """The database name of the mask, when it came from or was found in db."""
        return self._mask_name

    @property
    def linear(self) -> bool:
        """Whether data in this space is stored flat, one value per masked voxel."""
        return self._mask is not None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._mask is not None:
            return tuple(self._mask.shape)
        return _xfm_shape(self.subject, self.xfmname)

    def _with_mask(
        self, mask: npt.NDArray[np.bool_], spec: MaskSpec, name: Optional[str]
    ) -> "VolumeSpace":
        return VolumeSpace(
            self.subject, self.xfmname, spec, _resolved=mask, _mask_name=name
        )

    def resolve_for(self, data: Optional[npt.NDArray]) -> "VolumeSpace":
        if data is None:
            raise TypeError("Volumetric data cannot be None")
        if data.ndim not in (1, 2, 3, 4):
            raise ValueError("Invalid data shape")

        if data.ndim not in (1, 2):
            # Dense data: nothing to resolve, and a mask would be meaningless.
            return self if self._mask_spec is None else VolumeSpace(
                self.subject, self.xfmname
            )

        spec = self._mask_spec
        if self._mask is not None and spec is not None:
            return self
        if spec is None:
            # Guess the mask from the voxel count, as `_check_size` did.
            name, found = _find_mask(data.shape[-1], self.subject, self.xfmname)
            return self._with_mask(found, name, name)
        if isinstance(spec, np.ndarray):
            resolved = spec > 0
            return self._with_mask(resolved, resolved, None)
        return self._with_mask(_named_mask(self.subject, self.xfmname, spec), spec, spec)

    def validate(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        assert data is not None  # resolve_for rejected None already
        if self.linear:
            nvox = int(self._mask.sum())  # type: ignore[union-attr]
            if data.shape[-1] != nvox:
                raise ValueError(
                    "Masked data has %d voxels, but the mask selects %d"
                    % (data.shape[-1], nvox)
                )
            return data

        shape = data.shape[1:] if self.is_movie(data) else data.shape
        if tuple(shape) != _xfm_shape(self.subject, self.xfmname):
            raise ValueError(
                "Volumetric data (shape %s) is not the same shape as reference "
                "for transform (shape %s)"
                % (str(tuple(shape)), str(_xfm_shape(self.subject, self.xfmname)))
            )
        return data

    def is_movie(self, data: npt.NDArray) -> bool:
        return data.ndim in (2, 4)

    def to_dense(self, data: npt.NDArray) -> npt.NDArray:
        from .. import volume

        if self.linear:
            dense = volume.unmask(self._mask, data[:])
        else:
            dense = data[:]
        if not self.is_movie(data):
            dense = dense[np.newaxis]
        return dense

    def align(self, first: "BrainData", second: "BrainData") -> tuple[npt.NDArray, npt.NDArray]:
        """Stored arrays when both share one mask, unmasked volumes otherwise.

        A flat array's positions mean something only relative to its own mask, so
        two arrays under the *same* mask already line up -- and are far smaller.
        Under different masks, or with one flat and one dense, only the unmasked
        volumes are comparable. ``Volume2D.raw`` spelled this out inline.
        """
        s1, s2 = first.space, second.space
        assert isinstance(s1, VolumeSpace) and isinstance(s2, VolumeSpace)
        if s1.xfmname != s2.xfmname:
            raise ValueError(
                "Both Volumes must have same xfmname to generate single raw volume"
            )
        if s1.linear and s2.linear and s1 == s2:
            return first.data, second.data
        return first.dense, second.dense

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> "WebGLPayload":
        from ._webgl import MosaicTexture

        return MosaicTexture(data, raw=raw)

    def to_json(self) -> dict[str, Any]:
        xfm = db.get_xfm(self.subject, self.xfmname, "coord").xfm
        return {"xfm": [list(np.array(xfm).ravel())]}

    def describe_layout(self, data: npt.NDArray) -> dict[str, Any]:
        """The 3-D grid the mosaic tiles unpack back into."""
        return {"shape": self.shape}

    def write_hdf_attrs(self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset) -> None:
        if self._mask_spec is None:
            return

        from ._hdf import _hash, _hdf_write

        mask: Any = self._mask_spec
        if isinstance(self._mask_spec, np.ndarray):
            group = "/subjects/{subj}/transforms/{xfm}/masks/".format(
                subj=self.subject, xfm=self.xfmname
            )
            name = "__%s" % _hash(self._mask_spec)[:8]
            _hdf_write(h5, self._mask_spec, name=name, group=group)
            mask = name
        node.attrs["mask"] = mask

    @classmethod
    def from_hdf(
        cls, attrs: dict[str, Any], *, subject: str, xfmname: Optional[str], mask: MaskSpec
    ) -> Optional["VolumeSpace"]:
        if xfmname is None:
            return None
        return cls(subject, xfmname, mask)


@register_space
class SurfaceSpace(Space):
    """Per-vertex data on a subject's cortical surface.

    Claims any data node without a transform, which is how a file written before
    spaces existed -- and so carrying no discriminator -- is recognised.
    """

    fallback = True

    def _identity(self) -> tuple:
        return (self._subject,)

    @property
    def xfmname(self) -> None:
        """A surface space has no transform."""
        return None

    @property
    def llen(self) -> int:
        return _hemisphere_lengths(self.subject)[0]

    @property
    def rlen(self) -> int:
        return _hemisphere_lengths(self.subject)[1]

    @property
    def nverts(self) -> int:
        return self.llen + self.rlen

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.nverts,)

    def resolve_for(self, data: Optional[npt.NDArray]) -> "SurfaceSpace":
        return self

    def hemisphere_of(self, data: npt.NDArray) -> str:
        """Which hemispheres ``data`` covers: "left", "right" or "both".

        A fact about the array, not the space, which is why it is asked rather
        than stored -- and why it takes the *unpadded* array.
        """
        given = data.shape[-1]
        if given == self.llen:
            return "left"
        if given == self.rlen:
            return "right"
        if given == self.nverts:
            return "both"
        raise ValueError(
            "Invalid number of vertices for subject (given %d, should be %d for "
            "left hem, %d for right hem, or %d for both)"
            % (given, self.llen, self.rlen, self.nverts)
        )

    def validate(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Pad single-hemisphere data with zeros for the other hemisphere."""
        if data is None:
            return np.zeros((self.nverts,))

        hem = self.hemisphere_of(data)
        if hem == "both":
            return data

        movie = self.is_movie(data)
        other = list(data.shape)
        other[1 if movie else 0] = self.rlen if hem == "left" else self.llen
        pad = np.zeros(other, dtype=data.dtype)
        return np.hstack([data, pad] if hem == "left" else [pad, data])

    def is_movie(self, data: npt.NDArray) -> bool:
        return data.ndim > 1

    def to_dense(self, data: npt.NDArray) -> npt.NDArray:
        return data if self.is_movie(data) else data[np.newaxis]

    def split_hemispheres(self, data: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """``data`` cut at the hemisphere boundary.

        One implementation for what ``Vertex.left``/``right`` and
        ``VertexRGB.left``/``right`` each wrote out, the pair differing only in
        whether they indexed a frame axis.
        """
        # Axis 0 for a bare (nverts,) array, axis 1 for anything with a leading
        # frame axis -- which covers both a scalar movie (t, nverts) and the uint8
        # RGBA an RGB view samples to (t, nverts, 4).
        if data.ndim == 1:
            return data[: self.llen], data[self.llen :]
        return data[:, : self.llen], data[:, self.llen :]

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> "WebGLPayload":
        from ._webgl import VertexAttributes

        return VertexAttributes(data, raw=raw)

    def to_json(self) -> dict[str, Any]:
        return {}

    def describe_layout(self, data: npt.NDArray) -> dict[str, Any]:
        """Where the hemispheres meet, and how many frames there are."""
        return {
            "split": self.llen,
            "frames": data.shape[0] if self.is_movie(data) else 1,
        }

    def write_hdf_attrs(self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset) -> None:
        return None

    @classmethod
    def from_hdf(
        cls, attrs: dict[str, Any], *, subject: str, xfmname: Optional[str], mask: MaskSpec
    ) -> Optional["SurfaceSpace"]:
        if xfmname is not None:
            return None
        return cls(subject)
