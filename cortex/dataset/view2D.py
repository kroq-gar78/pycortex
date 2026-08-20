"""Two scalar channels through one 2D colormap.

``Dataview2D`` now calls ``Dataview.__init__`` instead of re-setting
``state``/``attrs``/``priority``/``description`` by hand, and asks
``space.align`` for the pair of arrays to colormap instead of comparing masks
inline. What is left in the two concrete classes is a constructor and a repr.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Generic, Optional, Union, cast

import h5py
import numpy as np
import numpy.typing as npt

from .. import options
from ._space import Space, SurfaceSpace, VolumeSpace
from .braindata import BrainData
from .views import (
    Dataview,
    DataviewJSON,
    DataviewScalar,
    ScalarT,
    Vertex,
    Volume,
    _dumps,
)

if TYPE_CHECKING:
    # Annotation only: `viewRGB` imports `_resolve_channels` from this module, so a
    # runtime import here would close the cycle.
    from .viewRGB import DataviewRGB

default_cmap2D = options.config.get("basic", "default_cmap2D")


def _resolve_channels(
    channels: list[Any],
    *,
    space_cls: type[Space],
    subject: Optional[str],
    spec: dict[str, Any],
    argnames: tuple[str, ...],
) -> tuple[Space, Optional[list[DataviewScalar]]]:
    """Validate a composite view's channel arguments and find their space.

    Returns the space to build in, plus the channels if they arrived as views and
    None if they arrived as raw arrays. The two forms cannot be mixed: either
    every channel is a scalar view and they agree on the space, or none is and the
    space's arguments must be given explicitly.

    ``Dataview2D`` and ``DataviewRGB`` each carried this, with the same checks in
    the same order and different wording. ``space1 == space2`` is what collapsed
    them: the subject/xfmname/mask comparisons were written out by hand.
    """
    first = channels[0]
    if isinstance(first, DataviewScalar):
        for name, chan in zip(argnames[1:], channels[1:]):
            if not isinstance(chan, DataviewScalar):
                raise TypeError(
                    "%s is not a %s object; if %s is one then all of them must be"
                    % (name, type(first).__name__, argnames[0])
                )
            if chan.subject != first.subject:
                raise TypeError("%s is from a different subject" % name)
        if subject is not None and first.subject != subject:
            raise ValueError(
                "Subject in channel objects (%r) is different than specified "
                "subject (%r)" % (first.subject, subject)
            )
        for key, value in spec.items():
            if value is not None and getattr(first.space, key, None) != value:
                raise ValueError(
                    "%s in channel objects (%r) is different than specified %s (%r)"
                    % (key, getattr(first.space, key, None), key, value)
                )
        return first.space, [c for c in channels]

    for name, chan in zip(argnames[1:], channels[1:]):
        if isinstance(chan, DataviewScalar):
            raise TypeError(
                "%s is a view object, so %s must be one as well" % (name, argnames[0])
            )
    return space_cls.from_spec(subject, **spec), None


class Dataview2D(Dataview, Generic[ScalarT]):
    """Abstract base class for 2-dimensional data views.

    Generic in the channel type, so ``Volume2D.dim1`` is a ``Volume`` and
    ``Vertex2D.dim1`` is a ``Vertex`` without either class re-declaring them. The
    channels are read-only, which is what makes treating this class as covariant
    in that type sound.
    """

    def __init__(
        self,
        dim1: ScalarT,
        dim2: ScalarT,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        state: Any = None,
        priority: int = 1,
        **attrs: Any,
    ) -> None:
        self._dim1 = dim1
        self._dim2 = dim2
        self.cmap = cmap or default_cmap2D
        # Each axis falls back to its own channel's range, which is resolved by
        # the time it gets here. This used to be done twice -- pre-resolved in the
        # subclasses, and again here with a *different* rule (vmin2 falling back
        # to vmin) that the pre-resolution meant could never fire.
        self.vmin = dim1.vmin if vmin is None else vmin
        self.vmax = dim1.vmax if vmax is None else vmax
        self.vmin2 = dim2.vmin if vmin2 is None else vmin2
        self.vmax2 = dim2.vmax if vmax2 is None else vmax2
        super().__init__(
            description=description, state=state, priority=priority, **attrs
        )

    @property
    def dim1(self) -> ScalarT:
        return self._dim1

    @property
    def dim2(self) -> ScalarT:
        return self._dim2

    @property
    def space(self) -> Space:
        return self.dim1.space

    def uniques(self, collapse: bool = False):
        yield self.dim1
        yield self.dim2

    def get_cmapdict(self) -> dict[str, Any]:
        """Colormap arguments for the *first* axis only.

        The second axis's range has no place in an ``imshow`` call; the 2D
        colorbar is built separately from all four bounds.
        """
        from .views import _lookup_cmap

        return dict(cmap=_lookup_cmap(self.cmap), vmin=self.vmin, vmax=self.vmax)

    def copy(self) -> "Dataview2D[ScalarT]":
        """A view of the same kind over the same two channels.

        The composite columns had no working ``copy()`` at all: they inherited
        ``Dataview.copy``, which splatted ``cmap=``/``vmin=``/``vmax=`` into
        ``self.__class__(...)``, and their constructors do not accept those in
        that position.
        """
        return type(self)(
            self.dim1,
            self.dim2,
            description=self.description,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            vmin2=self.vmin2,
            vmax2=self.vmax2,
            state=self.state,
            **self.attrs,
        )

    # ------------------------------------------------------------------
    # color
    # ------------------------------------------------------------------
    def _to_raw(
        self, data1: npt.NDArray, data2: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize

        cmapdir = options.config.get("webgl", "colormaps")
        cmap = plt.imread(os.path.join(cmapdir, "%s.png" % self.cmap))
        _warn_non_perceptually_uniform_colormap(self.cmap)

        d1 = np.clip(Normalize(self.vmin, self.vmax)(data1), 0, 1)
        d2 = np.clip(1 - Normalize(self.vmin2, self.vmax2)(data2), 0, 1)
        # NaNs interact badly with the conversion to uint32.
        dim1 = np.nan_to_num(np.round(d1 * (cmap.shape[1] - 1))).astype(np.uint32)
        dim2 = np.nan_to_num(np.round(d2 * (cmap.shape[0] - 1))).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # 0-255 rather than 0-1, to avoid problems in the RGB column downstream.
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = (channel.reshape(dim1.shape) for channel in colored.T)
        a = a.copy()
        a[np.logical_or(np.isnan(data1), np.isnan(data2))] = 0
        return r, g, b, a

    @property
    def raw(self) -> Dataview:
        """This view colormapped, as an RGB view of the same space.

        Asks ``space.align`` for a pair of arrays in which position *i* means the
        same place in both, which is the whole of what ``Volume2D.raw`` spelled
        out in fifteen lines of mask comparison and ``Vertex2D.raw`` in two.
        """
        first, second = self.space.align(self.dim1.braindata, self.dim2.braindata)
        r, g, b, a = self._to_raw(first, second)

        space = self.space
        wrap = type(space).scalar_view._from_parts
        # `alpha` stays an array: the RGB column wraps it with vmin=0/vmax=1, which
        # is what its NaN masking assigns into it.
        return type(space).rgb_view(
            wrap(BrainData(r, space)),
            wrap(BrainData(g, space)),
            wrap(BrainData(b, space)),
            alpha=self.attrs.get("alpha", a),
            description=self.description,
            state=self.state,
            priority=self.priority,
        )

    @property
    def dense(self) -> npt.NDArray[np.uint8]:
        """The array a renderer samples: this view colormapped to uint8 RGBA.

        A 2D view owns no array of its own, so unlike the other two columns this
        is derived rather than stored.
        """
        return cast("DataviewRGB[Any]", self.raw).dense

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        sdict = super().to_json(simple=simple)
        if simple:
            return sdict
        sdict.update(
            DataviewJSON(
                data=[[self.dim1.name, self.dim2.name]],
                cmap=[self.cmap],
                vmin=[[self.vmin, self.vmin2]],
                vmax=[[self.vmax, self.vmax2]],
            )
        )
        d1js = self.dim1.to_json()
        d2js = self.dim2.to_json()
        if "xfm" in d1js:
            sdict["xfm"] = [[d1js["xfm"][0], d2js["xfm"][0]]]
        return sdict

    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        view[2] = _dumps([self.cmap])
        view[3] = _dumps([[self.vmin, self.vmin2]])
        view[4] = _dumps([[self.vmax, self.vmax2]])

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self.dim1.braindata._write_hdf(h5)
        self.dim2.braindata._write_hdf(h5)
        return self._write_view_node(
            h5,
            name,
            [[self.dim1.name, self.dim2.name]],
            self._view_xfmname(),
        )

    def _view_xfmname(self) -> Optional[list[Any]]:
        """Slot 7. One transform name per dimension, for a space that has one."""
        if self.space.xfmname is None:
            return None
        return [[self.dim1.space.xfmname, self.dim2.space.xfmname]]


class Volume2D(Dataview2D[Volume]):
    """
    Contains two 3D volumes for simultaneous visualization. Includes information
    on how the volumes should be jointly colormapped.

    Parameters
    ----------
    dim1 : ndarray or Volume
        The first volume. Can be a 1D or 3D array (see Volume for details), or
        a Volume.
    dim2 : ndarray or Volume
        The second volume. Can be a 1D or 3D array (see Volume for details), or
        a Volume.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        dim1 must be a Volume from which the subject can be extracted.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        dim1 must be a Volume from which the transform can be extracted.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the
        ``default_cmap2D`` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. Defaults to dim1's own vmin.
    vmax : float, optional
        Maximum value in colormap for dim1. Defaults to dim1's own vmax.
    vmin2 : float, optional
        Minimum value in colormap for dim2. Defaults to dim2's own vmin.
    vmax2 : float, optional
        Maximum value in colormap for dim2. Defaults to dim2's own vmax.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        dim1: Union[npt.NDArray, Volume],
        dim2: Union[npt.NDArray, Volume],
        subject: Optional[str] = None,
        xfmname: Optional[str] = None,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        space, views = _resolve_channels(
            [dim1, dim2],
            space_cls=VolumeSpace,
            subject=subject,
            spec={"xfmname": xfmname},
            argnames=("dim1", "dim2"),
        )
        if views is None:
            chan1 = Volume(np.asarray(dim1), space.subject, space.xfmname, vmin=vmin, vmax=vmax)
            chan2 = Volume(np.asarray(dim2), space.subject, space.xfmname, vmin=vmin2, vmax=vmax2)
        else:
            # `_resolve_channels` is space-agnostic, so it can only promise the
            # base channel type. The cast is where this class states what its
            # generic parameter already fixed.
            chan1, chan2 = cast(tuple[Volume, Volume], tuple(views))

        super().__init__(
            chan1,
            chan2,
            description=description,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            **kwargs,
        )

    @property
    def xfmname(self) -> str:
        return self.dim1.space.xfmname

    @property
    def volume(self) -> npt.NDArray:
        """5D volume (t, z, y, x, rgba) of this view's colormapped data.

        ``Volume2D`` had no such accessor at all, though ``Vertex2D`` had
        ``vertices``: they were written per class rather than per column, and this
        one was missed.
        """
        return self.dense

    def __repr__(self) -> str:
        return "<2D volumetric data for (%s, %s)>" % (self.subject, self.xfmname)


class Vertex2D(Dataview2D[Vertex]):
    """
    Contains two vertex maps for simultaneous visualization. Includes information
    on how the maps should be jointly colormapped.

    Parameters
    ----------
    dim1 : ndarray or Vertex
        The first vertex map. Can be a 1D array (see Vertex for details), or
        a Vertex.
    dim2 : ndarray or Vertex
        The second vertex map. Can be a 1D array (see Vertex for details), or
        a Vertex.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        dim1 must be a Vertex from which the subject can be extracted.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the
        ``default_cmap2D`` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. Defaults to dim1's own vmin.
    vmax : float, optional
        Maximum value in colormap for dim1. Defaults to dim1's own vmax.
    vmin2 : float, optional
        Minimum value in colormap for dim2. Defaults to dim2's own vmin.
    vmax2 : float, optional
        Maximum value in colormap for dim2. Defaults to dim2's own vmax.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        dim1: Union[npt.NDArray, Vertex],
        dim2: Union[npt.NDArray, Vertex],
        subject: Optional[str] = None,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        space, views = _resolve_channels(
            [dim1, dim2],
            space_cls=SurfaceSpace,
            subject=subject,
            spec={},
            argnames=("dim1", "dim2"),
        )
        if views is None:
            chan1 = Vertex(np.asarray(dim1), space.subject, vmin=vmin, vmax=vmax)
            chan2 = Vertex(np.asarray(dim2), space.subject, vmin=vmin2, vmax=vmax2)
        else:
            chan1, chan2 = cast(tuple[Vertex, Vertex], tuple(views))

        super().__init__(
            chan1,
            chan2,
            description=description,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            **kwargs,
        )

    @property
    def vertices(self) -> npt.NDArray:
        """3D array (t, v, rgba) of this view's colormapped data."""
        return self.dense

    def __repr__(self) -> str:
        return "<2D vertex data for (%s)>" % self.subject


def _warn_non_perceptually_uniform_colormap(cmap: Any) -> None:
    mapping = {
        "BuOr_2D": "PU_BuOr_covar",
        "RdBu_covar": "PU_RdBu_covar",
        "RdBu_covar2": "PU_BuOr_covar",
        "RdBu_covar_alpha": "PU_RdBu_covar_alpha",
        "RdGn_covar": "PU_RdGn_covar",
        "hot_alpha": "fire_alpha",
    }
    if cmap in mapping:
        warnings.warn(
            "Colormap %r is not perceptually uniform. Consider using %r instead."
            % (cmap, mapping[cmap]),
            UserWarning,
        )
