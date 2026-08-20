"""Three scalar channels plus alpha, carrying their own colors.

``DataviewRGB`` now calls ``Dataview.__init__``, and reaches the array it ships
through ``space.to_dense`` rather than through a ``_cls`` class pointer used to
call unbound methods of whichever data class its subclass had in mind. That
pointer is what ``self._cls._write_hdf(self.red, h5)`` was, and it is gone.
"""

from __future__ import annotations

import colorsys
import warnings
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast

import h5py
import numpy as np
import numpy.typing as npt

from .. import options
from ._hdf import _hash
from ._space import Space, SurfaceSpace, VolumeSpace
from .braindata import BrainData
from .view2D import _resolve_channels
from .views import Dataview, DataviewJSON, DataviewScalar, ScalarT, Vertex, Volume

default_cmap = options.config.get("basic", "default_cmap")

ColorDtype = TypeVar("ColorDtype", int, float)
Color = tuple[ColorDtype, ColorDtype, ColorDtype]  # RGB color


class Colors:
    """
    Set of known colors
    """

    RoseRed: Color[int] = (237, 35, 96)
    LimeGreen: Color[int] = (141, 198, 63)
    SkyBlue: Color[int] = (0, 176, 218)
    DodgerBlue: Color[int] = (30, 144, 255)
    Red: Color[int] = (255, 000, 000)
    Green: Color[int] = (000, 255, 000)
    Blue: Color[int] = (000, 000, 255)


def RGB2HSV(color: Color | npt.NDArray) -> Color[float]:
    """
    Converts RGB to HS
    Parameters
    ----------
    color : tuple<uint8, uint8, uint8>
        RGB color value

    Returns
    -------
    tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    """
    hue, saturation, value = colorsys.rgb_to_hsv(
        color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    )
    hue *= 360
    return (int(hue), saturation, value)


def HSV2RGB(color: Color[float] | npt.NDArray) -> Color[int]:
    """
    Converts HSV to RGB

    Parameters
    ----------
    color : tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    Returns
    -------
    tuple<uint8, uint8, uint8>
        RGB color value
    """
    r, g, b = colorsys.hsv_to_rgb(color[0] / 360.0, color[1], color[2])
    return (int(r * 255), int(g * 255), int(b * 255))

def color_voxels(
    channel1: Union[npt.NDArray, DataviewScalar],
    channel2: Union[npt.NDArray, DataviewScalar],
    channel3: Union[npt.NDArray, DataviewScalar],
    channel1color: Color[int],
    channel2color: Color[int],
    channel3Color: Color[int],
    value_max: Optional[float],
    saturation_max: float,
    vmin: Optional[Union[float, tuple[float, float, float]]],
    vmax: Optional[Union[float, tuple[float, float, float]]],
    autorange: Literal['shared', 'individual'] = 'individual',
    alpha: Optional[Union[npt.NDArray, DataviewScalar]] = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Colors voxels in 3 color dimensions but not necessarily canonical red, green, and blue
    Parameters
    ----------
    channel1 : ndarray or scalar view
        voxel values for first channel
    channel2 : ndarray or scalar view
        voxel values for second channel
    channel3 : ndarray or scalar view
        voxel values for third channel
    channel1color : tuple<uint8, uint8, uint8>
        color in RGB for first channel
    channel2color : tuple<uint8, uint8, uint8>
        color in RGB for second channel
    channel3Color : tuple<uint8, uint8, uint8>
        color in RGB for third channel
    value_max : float, optional
        Maximum HSV value for voxel colors. If not given, will be the value of
        the average of the three channel colors.
    saturation_max : float [0, 1]
        Maximum HSV saturation for voxel colors.
    vmin : float or tuple of float, optional
        Lower bound(s) that map to 0 in each color channel. If a single float, the same lower bound
        is used for all three channels. If a tuple of three floats, each channel
        uses its respective value. If None, the lower bound is auto-determined
        based on ``autorange``.
    vmax : float or tuple of float, optional
        Upper bound(s) that map to 255 in each color channel. If a single float, the same upper bound
        is used for all three channels. If a tuple of three floats, each channel
        uses its respective value. If None, the upper bound is auto-determined
        based on ``autorange``.
    autorange : 'shared' or 'individual'
        How to auto-determine bounds when vmin or vmax is None. 'shared' computes
        the 1st and 99th percentile across all three channels combined. 'individual'
        computes per-channel 1st and 99th percentiles. Overridden when vmin and
        vmax are both provided.
    alpha : ndarray or scalar view, optional
        Alpha values for each voxel. If None, alpha is set to 1 for all voxels.

    Returns
    -------
    red : ndarray of channel1.shape
        uint8 array of red values
    green : ndarray of channel1.shape
        uint8 array of green values
    blue : ndarray of channel1.shape
        uint8 array of blue values
    alpha : ndarray
        If alpha=None, uint8 array of alpha values with alpha=1 for every voxel.
        Otherwise, the same alpha values that were passed in. Additionally,
        voxels with NaNs will have an alpha value of 0.
    """
    # normalize each channel to [0, 1]
    data1 = (
        channel1.data
        if isinstance(channel1, DataviewScalar)
        else channel1
    )
    data1 = data1.astype(float)
    data2 = (
        channel2.data
        if isinstance(channel2, DataviewScalar)
        else channel2
    )
    data2 = data2.astype(float)
    data3 = (
        channel3.data
        if isinstance(channel3, DataviewScalar)
        else channel3
    )
    data3 = data3.astype(float)

    if (data1.shape != data2.shape) or (data2.shape != data3.shape):
        raise ValueError("Volumes are of different shapes")

    # Create an alpha mask now, before casting nans to 0
    # Voxels with at least one channel equal to NaN will be masked out.
    mask = np.isnan(np.array([data1, data2, data3])).any(axis=0)
    # Now convert to NaNs to num for all channels
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)
    data3 = np.nan_to_num(data3)

    # Expand vmin/vmax to per-channel lists
    if isinstance(vmin, (int, float)):
        channel_vmins = [float(vmin), float(vmin), float(vmin)]
    elif vmin is not None:
        channel_vmins = [float(v) for v in vmin]
    else:
        channel_vmins = [None, None, None]

    if isinstance(vmax, (int, float)):
        channel_vmaxs = [float(vmax), float(vmax), float(vmax)]
    elif vmax is not None:
        channel_vmaxs = [float(v) for v in vmax]
    else:
        channel_vmaxs = [None, None, None]

    # Auto-determine any None bounds
    needs_auto_min = any(v is None for v in channel_vmins)
    needs_auto_max = any(v is None for v in channel_vmaxs)

    if needs_auto_min or needs_auto_max:
        if autorange == "shared":
            all_data = np.concatenate([data1.ravel(), data2.ravel(), data3.ravel()])
            shared_min = np.percentile(all_data, 1)
            shared_max = np.percentile(all_data, 99)
            channel_vmins = [shared_min if v is None else v for v in channel_vmins]
            channel_vmaxs = [shared_max if v is None else v for v in channel_vmaxs]
        elif autorange == "individual":
            for i, data in enumerate([data1, data2, data3]):
                if channel_vmins[i] is None:
                    channel_vmins[i] = np.percentile(data.ravel(), 1)
                if channel_vmaxs[i] is None:
                    channel_vmaxs[i] = np.percentile(data.ravel(), 99)
        else:
            raise ValueError("autorange must be 'shared' or 'individual'")

    normalized = []
    for channel, (data, channel_min, channel_max) in enumerate(
        zip([data1, data2, data3], channel_vmins, channel_vmaxs), start=1
    ):
        channel_range = channel_max - channel_min
        if channel_range == 0:
            warnings.warn(
                "Channel {} has no dynamic range (vmin == vmax) and will be zeroed out".format(
                    channel
                )
            )
            normalized.append(np.zeros_like(data))
        else:
            normalized.append((data - channel_min) / channel_range)
    data1, data2, data3 = normalized
    data1 = np.clip(data1, 0, 1)
    data2 = np.clip(data2, 0, 1)
    data3 = np.clip(data3, 0, 1)

    channel1color = np.array(channel1color)
    channel2color = np.array(channel2color)
    channel3Color = np.array(channel3Color)

    averageColor = (channel1color + channel2color + channel3Color) / 3

    if value_max is None:
        _, _, value = RGB2HSV(averageColor)
        value_max = value

    red = np.zeros_like(data1, np.uint8)
    green = np.zeros_like(data1, np.uint8)
    blue = np.zeros_like(data1, np.uint8)
    for i in range(data1.size):
        this_color = (
            data1.flat[i] * channel1color
            + data2.flat[i] * channel2color
            + data3.flat[i] * channel3Color
        )
        this_color /= 3.0
        if (value_max != 1.0) or (saturation_max != 1.0):
            hue, saturation, value = RGB2HSV(this_color)
            saturation /= saturation_max
            value /= value_max
            if saturation > 1:
                saturation = 1.0
            if value > 1:
                value = 1.0
            this_color = HSV2RGB((hue, saturation, value))
        red.flat[i] = this_color[0]
        green.flat[i] = this_color[1]
        blue.flat[i] = this_color[2]

    # Now make an alpha volume
    if alpha is None:
        alpha = np.ones_like(red, np.uint8) * 255
    alpha[mask] = 0 # TODO: this seems like an actual issue

    return red, green, blue, alpha


def _channel_array(channel: Any) -> npt.NDArray:
    return channel.data if isinstance(channel, DataviewScalar) else channel


class DataviewRGB(Dataview, Generic[ScalarT]):
    """Abstract base class for RGB data views.

    Generic in the channel type, so ``VolumeRGB.red`` is a ``Volume`` and
    ``VertexRGB.alpha`` is a ``Vertex``. That last one is why the TypeVar earns
    its keep: a property's return type cannot be narrowed by re-annotation, only
    by re-implementing the property, so ``alpha`` would otherwise exist twice.
    """

    def __init__(
        self,
        red: ScalarT,
        green: ScalarT,
        blue: ScalarT,
        alpha: Optional[Union[npt.NDArray, DataviewScalar]] = None,
        subject: Optional[str] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        **attrs: Any,
    ) -> None:
        self._red = red
        self._green = green
        self._blue = blue
        # `_alpha`, not the `alpha` property: the property *builds* one when none
        # was given, so testing it is always true. That is what made `uniques`
        # yield a synthesized alpha and `_write_hdf` save it -- and a saved
        # all-ones alpha reloads with vmin == vmax == 1, so normalizing it divides
        # by zero and the whole view comes back fully transparent.
        self._alpha = alpha

        if subject is not None and red.subject != subject:
            raise ValueError(
                "Subject in channel objects (%r) is different than specified "
                "subject (%r)" % (red.subject, subject)
            )
        if red.movie and not (
            red.data.shape[0] == green.data.shape[0] == blue.data.shape[0]
        ):
            raise ValueError(
                "For movie data, all three channels have to be the same length"
            )

        super().__init__(
            description=description, state=state, priority=priority, **attrs
        )

    @property
    def red(self) -> ScalarT:
        return self._red

    @property
    def green(self) -> ScalarT:
        return self._green

    @property
    def blue(self) -> ScalarT:
        return self._blue

    @property
    def space(self) -> Space:
        return self.red.space

    @property
    def movie(self) -> bool:
        return self.red.movie

    def uniques(self, collapse: bool = False):
        if collapse:
            yield self
        else:
            yield self.red
            yield self.green
            yield self.blue
            if self._alpha is not None:
                yield self.alpha

    def copy(self) -> "DataviewRGB[ScalarT]":
        return type(self)(
            self.red,
            self.green,
            self.blue,
            alpha=self._alpha,
            description=self.description,
            state=self.state,
            **self.attrs,
        )

    # ------------------------------------------------------------------
    # alpha
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> ScalarT:
        """The alpha channel, synthesized fully-opaque if none was given."""
        alpha = self._alpha
        if alpha is None:
            # Sized from the channel's *stored* array, which carries the frame
            # axis exactly when the channels do. Sizing from the sampled array's
            # vertex count alone made surface RGB *movies* unconstructible.
            alpha = np.ones(self.red.data.shape)
        if not isinstance(alpha, DataviewScalar):
            if alpha.dtype != np.uint8 and (alpha.min() < 0 or alpha.max() > 1):
                warnings.warn(
                    "Some alpha values are outside the range of [0, 1]. Consider "
                    "passing a view object as alpha with explicit vmin, vmax "
                    "keyword arguments.",
                    Warning,
                )
            alpha = type(self.space).scalar_view._from_parts(
                BrainData(alpha, self.space), vmin=0, vmax=1
            )

        # NaN in any channel means transparent, and uint8 cannot carry NaN, so
        # the mask captured before the conversion is applied here too.
        stacked = np.array([c.dense for c in (self.red, self.green, self.blue)])
        self._mask_alpha(alpha, np.isnan(stacked).any(axis=0))
        self._mask_alpha(alpha, self._nan_mask)
        return cast(ScalarT, alpha)

    @alpha.setter
    def alpha(self, alpha: Optional[Union[npt.NDArray, DataviewScalar]]) -> None:
        # Takes the *base* channel type, not ScalarT: mypy allows a covariant
        # TypeVar in __init__ parameters and in return position, but not in an
        # ordinary method parameter.
        self._alpha = alpha

    def _mask_alpha(
        self, alpha: DataviewScalar, mask: Optional[npt.NDArray[np.bool_]]
    ) -> None:
        """Force ``mask`` positions transparent, in whichever layout they arrive."""
        if mask is None:
            return
        if mask.shape == alpha.data.shape:
            alpha.data[mask] = alpha.vmin
        elif mask.shape == alpha.dense.shape:
            dense = alpha.dense
            dense[mask] = alpha.vmin
            alpha.data = dense if alpha.movie else dense[0]

    # ------------------------------------------------------------------
    # the sampled array
    # ------------------------------------------------------------------
    @property
    def dense(self) -> npt.NDArray[np.uint8]:
        """The four channels as one uint8 RGBA array, with a leading frame axis.

        One implementation for what ``VolumeRGB.volume`` and
        ``VertexRGB.vertices`` each wrote out: the same normalize-and-stack over
        the same four channels, differing only in which accessor they read and
        which axis order they transposed to.
        """
        channels = [
            _to_uint8(dv.dense, dv.vmin, dv.vmax)
            for dv in (self.red, self.green, self.blue, self.alpha)
        ]
        return np.moveaxis(np.array(channels), 0, -1)

    @property
    def name(self) -> str:
        """Content hash of the RGBA array. Only ever a browser key.

        The channels are what become HDF nodes, each under its own name.
        """
        return "__%s" % _hash(self.dense)[:16]

    def __hash__(self) -> int:
        return hash(_hash(self.dense))

    @property
    def raw(self) -> "DataviewRGB":
        return self

    def get_cmapdict(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        sdict = super().to_json(simple=simple)
        if simple:
            sdict.update(
                DataviewJSON(name=self.name, subject=self.subject, min=0, max=255)
            )
            sdict.update(cast(DataviewJSON, self.space.describe_layout(self.dense)))
        else:
            sdict.update(
                DataviewJSON(
                    data=[self.name], cmap=[default_cmap], vmin=[0], vmax=[255]
                )
            )
            sdict.update(cast(DataviewJSON, self.space.to_json()))
        return sdict

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        for channel in (self.red, self.green, self.blue):
            channel.braindata._write_hdf(h5)

        alpha = None
        if self._alpha is not None:
            self.alpha.braindata._write_hdf(h5)
            alpha = self.alpha.name

        return self._write_view_node(
            h5,
            name,
            [[self.red.name, self.green.name, self.blue.name, alpha]],
            self.space.view_xfmname,
        )


def _to_uint8(
    data: npt.NDArray, vmin: Optional[float], vmax: Optional[float]
) -> npt.NDArray[np.uint8]:
    """One channel scaled into 0-255, or passed through if already uint8."""
    if data.dtype == np.uint8:
        return data.copy()

    scaled = data.astype("float32", copy=True)
    if vmin is None:
        if scaled.min() < 0:
            scaled -= scaled.min()
    else:
        scaled -= vmin

    if vmax is None:
        if scaled.max() > 1:
            scaled /= scaled.max()
    else:
        scaled /= vmax - vmin
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _resolve_rgb_channels(
    channels: tuple[Any, Any, Any],
    *,
    space_cls: type[Space],
    subject: Optional[str],
    spec: dict[str, Any],
    colors: tuple[Color[int], Color[int], Color[int]],
    max_color_value: Optional[float],
    max_color_saturation: float,
    vmin: Optional[Union[float, tuple]],
    vmax: Optional[Union[float, tuple]],
    autorange: str,
    alpha: Any,
) -> tuple[tuple[DataviewScalar, DataviewScalar, DataviewScalar], Any]:
    """The three channels as scalar views, remapping colors if asked.

    Both concrete RGB classes ran this same forty-line decision twice each, once
    for view arguments and once for raw arrays, differing only in which class they
    named. The space answers that now.
    """
    space, views = _resolve_channels(
        list(channels),
        space_cls=space_cls,
        subject=subject,
        spec=spec,
        argnames=("channel1", "channel2", "channel3"),
    )
    wrap = type(space).scalar_view._from_parts

    passthrough = (
        colors == (Colors.Red, Colors.Green, Colors.Blue)
        and vmin is None
        and vmax is None
        and autorange == "individual"
    )
    if passthrough:
        if views is not None:
            return (views[0], views[1], views[2]), alpha
        for name, chan in zip(("channel2", "channel3"), channels[1:]):
            if not isinstance(chan, np.ndarray):
                raise TypeError(
                    "Data channels must be numpy arrays if channel1 is a numpy array"
                )
        return (
            wrap(BrainData(channels[0], space)),
            wrap(BrainData(channels[1], space)),
            wrap(BrainData(channels[2], space)),
        ), alpha

    if views is None:
        for chan in channels[1:]:
            if not isinstance(chan, np.ndarray):
                raise TypeError(
                    "Data channels must be numpy arrays if channel1 is a numpy array"
                )
    red, green, blue, alpha = color_voxels(
        channels[0],
        channels[1],
        channels[2],
        colors[0],
        colors[1],
        colors[2],
        max_color_value,
        max_color_saturation,
        vmin,
        vmax,
        autorange,
        alpha=alpha,
    )
    return (
        wrap(BrainData(red, space)),
        wrap(BrainData(green, space)),
        wrap(BrainData(blue, space)),
    ), alpha


class VolumeRGB(DataviewRGB[Volume]):
    """
    Contains RGB (or RGBA) colors for each voxel in a volumetric dataset.
    Includes information about the subject and transform for the data.

    Three data channels are mapped into a 3D color set. By default the data
    channels are mapped on to red, green, and blue. They can also be mapped to
    be different colors as specified, and then linearly combined.

    Parameters
    ----------
    channel1 : ndarray or Volume
        Array or Volume for the first data channel for each voxel.
    channel2 : ndarray or Volume
        Array or Volume for the second data channel for each voxel.
    channel3 : ndarray or Volume
        Array or Volume for the third data channel for each voxel.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        channel1 must be a Volume from which the subject can be extracted.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        channel1 must be a Volume from which the transform can be extracted.
    alpha : ndarray or Volume, optional
        Alpha component of the color for each voxel. If None, all voxels are
        assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        Viewer state to restore with this dataset.
    channel1color : tuple<uint8, uint8, uint8>
        RGB color to use for the first data channel.
    channel2color : tuple<uint8, uint8, uint8>
        RGB color to use for the second data channel.
    channel3color : tuple<uint8, uint8, uint8>
        RGB color to use for the third data channel.
    max_color_value : float [0, 1], optional
        Maximum HSV value for voxel colors. If not given, will be the value of
        the average of the three channel colors.
    max_color_saturation : float [0, 1]
        Maximum HSV saturation for voxel colors.
    vmin : float or tuple of float, optional
        Lower bound(s) that map to 0 in each color channel. A single float is
        used for all three; a tuple of three gives one per channel. If None, the
        bound is auto-determined per ``autorange``.
    vmax : float or tuple of float, optional
        Upper bound(s) that map to 255 in each color channel, as for ``vmin``.
    autorange : 'shared' or 'individual'
        How to auto-determine bounds when vmin or vmax is None. 'shared' computes
        the 1st and 99th percentile across all three channels combined;
        'individual' computes per-channel percentiles. Default 'individual'.
    priority : int, optional
        Priority for display ordering. Default is 1.
    """

    def __init__(
        self,
        channel1: Union[npt.NDArray, Volume],
        channel2: Union[npt.NDArray, Volume],
        channel3: Union[npt.NDArray, Volume],
        subject: Optional[str] = None,
        xfmname: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, Volume]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple]] = None,
        vmax: Optional[Union[float, tuple]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
    ) -> None:
        chans, resolved_alpha = _resolve_rgb_channels(
            (channel1, channel2, channel3),
            space_cls=VolumeSpace,
            subject=subject,
            spec={"xfmname": xfmname},
            colors=(tuple(channel1color), tuple(channel2color), tuple(channel3color)),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin,
            vmax=vmax,
            autorange=autorange,
            alpha=alpha,
        )
        red, green, blue = cast(tuple[Volume, Volume, Volume], chans)
        super().__init__(
            red,
            green,
            blue,
            alpha=resolved_alpha,
            subject=subject,
            description=description,
            state=state,
            priority=priority,
        )

    @property
    def space(self) -> VolumeSpace:
        return cast(VolumeSpace, self.red.space)

    @property
    def xfmname(self) -> str:
        return self.space.xfmname

    @property
    def volume(self) -> npt.NDArray[np.uint8]:
        """5D volume (t, z, y, x, rgba) of 8-bit colors."""
        return self.dense

    def __repr__(self) -> str:
        return "<RGB volumetric data for (%s, %s)>" % (self.subject, self.xfmname)


class VertexRGB(DataviewRGB[Vertex]):
    """
    Contains RGB (or RGBA) colors for each vertex in a surface dataset.
    Includes information about the subject.

    Parameters
    ----------
    red : ndarray or Vertex
        Array or Vertex for the first data channel for each vertex.
    green : ndarray or Vertex
        Array or Vertex for the second data channel for each vertex.
    blue : ndarray or Vertex
        Array or Vertex for the third data channel for each vertex.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        red must be a Vertex from which the subject can be extracted.
    alpha : ndarray or Vertex, optional
        Alpha component of the color for each vertex. If None, all vertices are
        assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        Viewer state to restore with this dataset.
    channel1color : tuple<uint8, uint8, uint8>
        RGB color to use for the first data channel.
    channel2color : tuple<uint8, uint8, uint8>
        RGB color to use for the second data channel.
    channel3color : tuple<uint8, uint8, uint8>
        RGB color to use for the third data channel.
    max_color_value : float [0, 1], optional
        Maximum HSV value for vertex colors. If not given, will be the value of
        the average of the three channel colors.
    max_color_saturation : float [0, 1]
        Maximum HSV saturation for vertex colors.
    vmin : float or tuple of float, optional
        Lower bound(s) that map to 0 in each color channel. A single float is
        used for all three; a tuple of three gives one per channel. If None, the
        bound is auto-determined per ``autorange``.
    vmax : float or tuple of float, optional
        Upper bound(s) that map to 255 in each color channel, as for ``vmin``.
    autorange : 'shared' or 'individual'
        How to auto-determine bounds when vmin or vmax is None. 'shared' computes
        the 1st and 99th percentile across all three channels combined;
        'individual' computes per-channel percentiles. Default 'individual'.
    priority : int, optional
        Priority for display ordering. Default is 1.
    """

    def __init__(
        self,
        red: Union[npt.NDArray, Vertex],
        green: Union[npt.NDArray, Vertex],
        blue: Union[npt.NDArray, Vertex],
        subject: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, Vertex]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple]] = None,
        vmax: Optional[Union[float, tuple]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
    ) -> None:
        chans, resolved_alpha = _resolve_rgb_channels(
            (red, green, blue),
            space_cls=SurfaceSpace,
            subject=subject,
            spec={},
            colors=(tuple(channel1color), tuple(channel2color), tuple(channel3color)),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin,
            vmax=vmax,
            autorange=autorange,
            alpha=alpha,
        )
        r, g, b = cast(tuple[Vertex, Vertex, Vertex], chans)
        super().__init__(
            r,
            g,
            b,
            alpha=resolved_alpha,
            subject=subject,
            description=description,
            state=state,
            priority=priority,
        )

    @property
    def space(self) -> SurfaceSpace:
        return cast(SurfaceSpace, self.red.space)

    @property
    def vertices(self) -> npt.NDArray[np.uint8]:
        """3D array (t, v, rgba) of 8-bit colors."""
        return self.dense

    @property
    def left(self) -> npt.NDArray[np.uint8]:
        return self.space.split_hemispheres(self.vertices)[0]

    @property
    def right(self) -> npt.NDArray[np.uint8]:
        return self.space.split_hemispheres(self.vertices)[1]

    def __repr__(self) -> str:
        return "<RGB vertex data for (%s)>" % self.subject
