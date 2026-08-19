"""This module defines a class Package which is used by webgl to encode pycortex datasets into json objects.
The general structure of the object that's transmitted looks like this:

dict(
    views = [ dict(name="proper name", cmap=cmap, vmin=vmin, vmax=vmax, data=["__braindata_name"]) ],
    data  = dict(__braindata_name=dict(subject=subject, min=min, max=max)),
    images=(__braindata_name=["img1.png", "img2.png"]),
)
"""

import os
import json
import numpy as np

from .. import dataset
from typing import Any, Optional, TypedDict

class PackageMetadata(TypedDict):
    views: list[dataset.DataviewJSON]
    data: dict[str, dataset.DataviewJSON]
    images: dict[str, list[str]]

# TODO: How to package multiviews?
class Package(object):
    """Package the data into a form usable by javascript"""

    def __init__(self, data):
        self.dataset = dataset.normalize(data)
        self.uniques = list(data.uniques(collapse=True))
        self.subjects: set[str] = set()

        self.brains: dict[str, dataset.DataviewJSON] = dict()
        # Two-phase, which is why this is not `list[bytes]`: the mosaicked-texture
        # encoding finishes as PNG bytes here, but the per-vertex one leaves an
        # array in place and only `reorder` turns it into `.npy` bytes, since it
        # cannot serialise before the CTM's vertex order is known.
        self.images: dict[str, list[Any]] = dict()
        # Kept so `reorder` asks the same encoding that produced the frames what
        # to do with them, rather than re-deriving it from the view's class.
        self._payloads: dict[str, Any] = dict()
        for brain in self.uniques:
            name = brain.name
            self.subjects.add(brain.subject)
            self.brains[name] = brain.to_json(simple=True)
            # Two questions, both answered by the view rather than by its class:
            # `dense` is the array to ship, and `space.pack_for_webgl` is how it
            # reaches the browser. This module used to answer the second itself,
            # in three `isinstance(brain, (Vertex, VertexRGB))` forks -- the
            # premultiplied-alpha asymmetry, the packing, and the vertex
            # reordering -- so one fact was restated once per consequence, and a
            # space this module had not heard of took whichever `else` came first.
            payload = brain.space.pack_for_webgl(
                brain.dense, raw=isinstance(brain, dataset.DataviewRGB)
            )
            self._payloads[name] = payload
            self.images[name] = payload.frames
            self.brains[name].update(payload.describe())

    @property
    def views(self) -> list[dataset.DataviewJSON]:
        metadata = []
        for name, view in self.dataset:
            meta = view.to_json(simple=False)
            meta["name"] = name
            if "stim" in meta["attrs"]:
                meta["attrs"]["stim"] = os.path.split(meta["attrs"]["stim"])[1]
            metadata.append(meta)
        return metadata

    def reorder(self, subjects: dict[str, str]) -> None:
        indices = dict(
            (k, np.load(os.path.splitext(v)[0] + ".npz")) for k, v in subjects.items()
        )
        for brain in self.uniques:
            # Whether permuting applies is the encoding's business, and only the
            # per-vertex one says yes -- the default `reorder` returns the frames
            # untouched without reading the index, so a mosaicked view does not
            # decompress an index array it has no use for.
            name = brain.name
            self.images[name] = self._payloads[name].reorder(
                self.images[name], indices[brain.subject]
            )
        for npz in indices.values():
            npz.close()

    # TODO: submap?
    def metadata(self, submap: Optional[dict[str, str]]=None, **kwargs) -> PackageMetadata:
        if submap is not None:
            for data in self.brains.values():
                data["subject"] = submap[data["subject"]]
        return PackageMetadata(
            views=self.views, data=self.brains, images=self.image_names(**kwargs)
        )

    def image_names(self, fmt: str="/data/{name}/{frame}/") -> dict[str, list[str]]:
        names: dict[str, list[str]] = dict()
        for name, imgs in self.images.items():
            names[name] = [fmt.format(name=name, frame=i) for i in range(len(imgs))]
        return names
