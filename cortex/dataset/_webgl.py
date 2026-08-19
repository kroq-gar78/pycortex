"""The two webgl wire encodings, one class each.

A compatibility surface, not a place to tidy up. Everything here is read by
``webgl/resources/js/dataset.js``, which dispatches on the *shape* of what it
receives -- ``mosaic === undefined`` means per-vertex attributes, anything else
means a mosaicked texture -- so a change in these bytes breaks the viewer
silently rather than noisily.

A space says which encoding its arrays use by returning one of these from
``Space.pack_for_webgl``. That is the whole extension point: ``webgl/data.py``
used to decide it itself, in three ``isinstance(brain, (Vertex, VertexRGB))``
forks -- the premultiplied-alpha asymmetry, the packing, and the vertex
reordering -- so one fact was restated once per consequence, and a space this
module had not heard of took whichever ``else`` came first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt


class WebGLPayload(ABC):
    """One array, encoded for the browser.

    Subclasses encode in ``__init__``: by the time a payload exists,
    :attr:`frames` is what will be served.
    """

    #: What gets served, one entry per frame. PNG bytes for a mosaicked texture;
    #: for per-vertex attributes a single array that only becomes ``.npy`` bytes
    #: in :meth:`reorder`, which cannot run until the CTM's vertex order is known.
    frames: list[Any]

    #: Whether the array is 4-channel uint8 from an RGB view rather than scalar
    #: floats. Shipped as the ``raw`` JSON key, and for per-vertex attributes it
    #: decides whether reordering has to index past a trailing channel axis.
    raw: bool

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """The JSON keys this encoding contributes to the view's data record.

        Keys must be *absent* rather than null when they do not apply:
        ``dataset.js`` selects the texture path by testing ``mosaic === undefined``.
        """

    def reorder(self, frames: list[Any], vertex_index: Any) -> list[Any]:
        """Frames permuted into the CTM's vertex order, if that applies.

        A no-op by default, and ``vertex_index`` -- the opened ``.npz`` of index
        arrays -- is deliberately left unread, so an encoding with no use for it
        never decompresses one.
        """
        return frames


class MosaicTexture(WebGLPayload):
    """Volumetric encoding: each frame tiled into one PNG, sampled as a texture.

    Alpha is **not** premultiplied here. Three.js sets ``tex.premultiplyAlpha``
    on upload and ``UNPACK_PREMULTIPLY_ALPHA_WEBGL`` does it once on the GPU, so
    doing it in Python as well would double-attenuate.
    """

    def __init__(self, data: npt.NDArray, *, raw: bool) -> None:
        # Deferred: `cortex.volume` imports `cortex.dataset` at module level.
        from ..volume import mosaic

        self.raw = raw
        data = data.astype(np.uint8 if raw else np.float32)
        tiles = [mosaic(frame, show=False) for frame in data]
        shapes = {shape for _, shape in tiles}
        if len(shapes) != 1:
            raise ValueError(
                "Frames of one view tiled to different mosaic shapes: %r" % (shapes,)
            )
        self.mosaic: tuple[int, int] = tiles[0][1]
        self.frames = [pack_png(tile) for tile, _ in tiles]

    def describe(self) -> dict[str, Any]:
        return {"raw": self.raw, "mosaic": self.mosaic}


class VertexAttributes(WebGLPayload):
    """Surface encoding: raw per-vertex attributes, served as ``.npy`` bytes.

    Alpha **is** premultiplied here, because these bytes reach the shader as
    vertex attributes and nothing else premultiplies them: the fragment shader
    composites with ``gl_FragColor = vColor + (1-a)*bg``, which is only correct
    for premultiplied colour (issue #631). The ``vertices``/``volume``
    properties stay straight-alpha, so the matplotlib path keeps working.
    """

    def __init__(self, data: npt.NDArray, *, raw: bool) -> None:
        self.raw = raw
        # `astype` copies even when the dtype already matches, so premultiplying
        # below writes into an array nothing else holds.
        data = data.astype(np.uint8 if raw else np.float32)
        if raw:
            alpha = data[..., 3:4].astype(np.float32) / 255.0
            data[..., :3] = np.round(data[..., :3].astype(np.float32) * alpha)
        self.frames = [data]

    def describe(self) -> dict[str, Any]:
        return {"raw": self.raw}

    def reorder(self, frames: list[Any], vertex_index: Any) -> list[Any]:
        index = vertex_index["index"]
        data = np.array(frames)[0]
        data = data[..., index, :] if self.raw else data[..., index]
        buf = BytesIO()
        np.save(buf, np.ascontiguousarray(data))
        buf.seek(0)
        return [buf.read()]


def pack_png(tile: npt.NDArray) -> bytes:
    """One mosaic tile as PNG bytes."""
    from PIL import Image

    if tile.dtype not in (np.float32, np.uint8):
        raise TypeError("Cannot pack %s as an RGBA PNG" % tile.dtype)

    y, x = tile.shape[:2]
    im = Image.frombuffer(
        "RGBA", (x, y), np.ascontiguousarray(tile).tobytes(), "raw", "RGBA", 0, 1
    )
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
