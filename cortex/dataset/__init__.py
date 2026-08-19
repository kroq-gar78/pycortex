"""Brain data in volumetric or surface form, and how it becomes color.

Three layers, related by composition only (see DATASET_REFACTOR.md):

    Dataview      how values become color   -- cmap, vmin/vmax, description, state
       holds BrainData    the values
          holds Space     the geometry

The six public classes are unchanged and are thin: they build a space, build a
:class:`BrainData`, and hand both to their column. The *space* is the open axis --
adding a kind of brain data means adding a :class:`Space`, not reimplementing
colormapping, HDF and JSON three more times -- and consumers dispatch on it
rather than on which of the six classes they were handed.
"""

from __future__ import annotations

from ._space import (
    Space,
    SurfaceSpace,
    VolumeSpace,
    detect_space,
    register_space,
    registered_spaces,
)
from ._webgl import MosaicTexture, VertexAttributes, WebGLPayload
from .braindata import BrainData, VertexData, VolumeData
from .views import (
    Colors,
    Dataview,
    Dataview2D,
    DataviewJSON,
    DataviewRGB,
    DataviewScalar,
    Vertex,
    Vertex2D,
    VertexRGB,
    Volume,
    Volume2D,
    VolumeRGB,
    _from_hdf_data,
)
from .dataset import Dataset, DatasetLike, normalize

__all__ = [
    # the six public view classes
    "Volume",
    "Vertex",
    "Volume2D",
    "Vertex2D",
    "VolumeRGB",
    "VertexRGB",
    # containers and helpers
    "Dataset",
    "DatasetLike",
    "Colors",
    "normalize",
    "DataviewJSON",
    # the three columns
    "Dataview",
    "DataviewScalar",
    "Dataview2D",
    "DataviewRGB",
    # values, and the names it used to be split under
    "BrainData",
    "VolumeData",
    "VertexData",
    # spaces -- the open axis
    "Space",
    "VolumeSpace",
    "SurfaceSpace",
    "register_space",
    "registered_spaces",
    "detect_space",
    # the two webgl wire encodings a space can pack its arrays into
    "WebGLPayload",
    "MosaicTexture",
    "VertexAttributes",
]
