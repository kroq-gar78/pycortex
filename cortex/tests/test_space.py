"""Tests for the space axis: the part of the restructure that has to pay off.

Everything here is about whether adding a kind of brain data is an addition or a
rewrite. Two halves: the value semantics that let a space be shared and compared,
and an end-to-end check that a space this package has never seen reaches every
consumer without one of them being edited.
"""

import os
import tempfile

import cortex
import numpy as np
import pytest

from cortex import dataset
from cortex.dataset import Space, SurfaceSpace, VolumeSpace

subj, xfmname, nverts, volshape = "S1", "fullhead", 304380, (31, 100, 100)


# ----------------------------------------------------------------------
# spaces are values
# ----------------------------------------------------------------------
def test_two_spaces_over_the_same_geometry_are_equal_and_hash_alike():
    """The comparison that replaced the hand-written subject/xfmname/mask checks.

    ``Volume2D.raw``, ``VolumeRGB.__init__`` and ``Vertex2D.__init__`` each spelled
    out their own version of "are these two the same place", and none of the three
    agreed on what to compare. Being a value means the space answers it once.
    """
    a = VolumeSpace(subj, xfmname)
    b = VolumeSpace(subj, xfmname)
    assert a == b and hash(a) == hash(b)
    assert VolumeSpace(subj, xfmname) != VolumeSpace(subj, "identity")

    # across kinds, and against a non-space, without raising
    assert SurfaceSpace(subj) == SurfaceSpace(subj)
    assert VolumeSpace(subj, xfmname) != SurfaceSpace(subj)
    assert SurfaceSpace(subj) != "not a space"

    # usable as a dict key, which is what "value" is for
    assert len({VolumeSpace(subj, xfmname), VolumeSpace(subj, xfmname)}) == 1


def test_a_masked_space_differs_from_an_unmasked_one_over_the_same_grid():
    """The mask is part of the identity, since it decides what a position means."""
    mask = cortex.db.get_mask(subj, xfmname, "thick")
    flat = cortex.Volume(np.random.randn(int(mask.sum())), subj, xfmname, mask="thick")
    dense = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    assert flat.space != dense.space
    assert flat.space.linear and not dense.space.linear
    # and two flat views under the same mask *are* in the same space, which is what
    # lets a 2D view colormap their stored arrays directly
    other = cortex.Volume(np.random.randn(int(mask.sum())), subj, xfmname, mask="thick")
    assert flat.space == other.space


def test_resolving_a_space_against_data_does_not_mutate_it():
    """``resolve_for`` returns a new value. Immutability is what makes sharing safe.

    Sharing was the point: the space carries whatever the geometry lookup produced,
    so ``copy()`` does not re-resolve a mask, and a mask learned from one array
    cannot leak onto the space another view was built from.
    """
    mask = cortex.db.get_mask(subj, xfmname, "thick")
    unresolved = VolumeSpace(subj, xfmname)
    resolved = unresolved.resolve_for(np.random.randn(int(mask.sum())))

    assert resolved is not unresolved
    assert resolved.linear and not unresolved.linear
    assert unresolved.mask is None


def test_a_copy_reuses_the_space_rather_than_rebuilding_it():
    """What made ``copy()`` collapse from one implementation per class to one.

    ``VolumeData.copy`` passed ``subject``, ``xfmname`` and a private ``_mask`` so
    the mask would not be re-resolved; ``VertexData.copy`` passed ``subject``
    alone. Carrying the space over is the same optimization, without the argument
    list.
    """
    mask = cortex.db.get_mask(subj, xfmname, "thick")
    view = cortex.Volume(np.random.randn(int(mask.sum())), subj, xfmname, mask="thick")
    copied = view.copy(np.random.randn(int(mask.sum())))

    assert copied.space is view.space
    assert copied.space.mask_name == "thick"
    assert type(copied) is type(view)


def test_the_database_is_consulted_once_per_geometry_not_once_per_view():
    """``db.get_surf`` ran for every ``Vertex`` built, for two integers.

    They depend on nothing but the subject, so they are cached on the subject. This
    asserts the cache is *used*, not merely present: a thousand views must not be a
    thousand surface loads.
    """
    from cortex.dataset import _space

    _space._hemisphere_lengths.cache_clear()
    data = np.random.randn(nverts)
    for _ in range(5):
        cortex.Vertex(data, subj)

    info = _space._hemisphere_lengths.cache_info()
    assert info.misses == 1
    assert info.hits >= 4


# ----------------------------------------------------------------------
# a space this package has never seen
# ----------------------------------------------------------------------
def _third_space():
    """A space over ten unrelated values, and the three views of it.

    Deliberately neither volumetric nor surface: it samples through no transform
    and has no hemispheres, so nothing it does can accidentally take a built-in
    path. Written the way the docs prescribe, so it is also a check that the
    prescription works.
    """
    nthings = 10

    class ThingSpace(Space):
        spec_keys = ("group",)

        def __init__(self, subject, group="a"):
            super().__init__(subject)
            self._group = group

        def _identity(self):
            return (self.subject, self._group)

        @property
        def group(self):
            return self._group

        @property
        def xfmname(self):
            return None

        @property
        def shape(self):
            return (nthings,)

        def resolve_for(self, data):
            return self

        def validate(self, data):
            if data is None:
                data = np.zeros((nthings,))
            if data.shape[-1] != nthings:
                raise ValueError("expected %d things" % nthings)
            return data

        def is_movie(self, data):
            return data.ndim > 1

        def to_dense(self, data):
            return data if self.is_movie(data) else data[np.newaxis]

        def to_json(self):
            return {}

        def describe_layout(self, data):
            return {"things": nthings}

        def write_hdf_attrs(self, h5, node):
            node.attrs["group"] = self._group

        @classmethod
        def from_hdf(cls, attrs, *, subject, xfmname, mask):
            if "group" not in attrs:
                return None
            return cls(subject, attrs["group"])

    class Thing(dataset.DataviewScalar):
        def __init__(self, data, subject, group="a", **kwargs):
            super().__init__(dataset.BrainData(data, ThingSpace(subject, group)), **kwargs)

    class Thing2D(dataset.Dataview2D):
        def __init__(self, dim1, dim2, subject=None, group=None, **kwargs):
            if not isinstance(dim1, Thing):
                dim1 = Thing(dim1, subject, group)
                dim2 = Thing(dim2, subject, group)
            super().__init__(dim1, dim2, **kwargs)

    class ThingRGB(dataset.DataviewRGB):
        def __init__(self, r, g, b, subject=None, group=None, alpha=None, **kwargs):
            if not isinstance(r, Thing):
                r, g, b = (Thing(c, subject, group) for c in (r, g, b))
            super().__init__(r, g, b, alpha=alpha, **kwargs)

    ThingSpace.scalar_view = Thing
    ThingSpace.twod_view = Thing2D
    ThingSpace.rgb_view = ThingRGB
    return ThingSpace, Thing, Thing2D, ThingRGB, nthings


def test_a_third_space_gets_the_three_columns_for_free():
    """The claim the whole restructure is for: a space, and nothing reimplemented.

    None of colormapping, ``.raw``, ``uniques``, ``to_json``, the arithmetic
    operators or the frame-axis handling is written by the space below, and all of
    it works.
    """
    _, Thing, Thing2D, ThingRGB, n = _third_space()
    arr = np.random.randn(n)

    view = Thing(arr, subj, cmap="hot")
    assert view.subject == subj
    assert view.vmin is not None and view.vmax is not None  # resolved by the column
    assert view.dense.shape == (1, n)
    assert (view + 1).data[0] == pytest.approx(arr[0] + 1)
    assert view.name.startswith("__")

    # `.raw` names no concrete class, so it produces this space's RGB view
    raw = view.raw
    assert isinstance(raw, ThingRGB)
    assert raw.dense.shape == (1, n, 4) and raw.dense.dtype == np.uint8

    twod = Thing2D(arr, arr * 2, subj, "a")
    assert isinstance(twod.raw, ThingRGB)
    assert twod.dense.shape == (1, n, 4)

    rgb = ThingRGB(arr, arr * 2, arr * 3, subj, "a")
    assert rgb.dense.shape == (1, n, 4)

    # movies, which the frame axis handling is the only reason to get right
    movie = Thing(np.random.randn(3, n), subj)
    assert movie.movie and movie.dense.shape == (3, n)
    assert ThingRGB(*([np.random.randn(3, n)] * 3), subj, "a").dense.shape == (3, n, 4)

    # and the layout keys the space contributes reach the browser payload
    assert view.to_json(simple=True)["things"] == n


def test_a_third_space_round_trips_through_hdf():
    """No factory edits: detection walks the registry and the space rebuilds itself."""
    from cortex.dataset import _space as space_mod

    ThingSpace, Thing, Thing2D, ThingRGB, n = _third_space()
    saved = list(space_mod._SPACES)
    path = tempfile.mktemp(suffix=".hdf")
    try:
        space_mod.register_space(ThingSpace)
        for name, view in (
            ("scalar", Thing(np.random.randn(n), subj, "b")),
            ("twod", Thing2D(np.random.randn(n), np.random.randn(n), subj, "b")),
            ("rgb", ThingRGB(*([np.random.randn(n)] * 3), subj, "b")),
        ):
            if os.path.exists(path):
                os.unlink(path)
            dataset.Dataset(**{name: view}).save(path)
            reloaded = cortex.load(path)[name]
            assert type(reloaded) is type(view), name
            assert isinstance(reloaded.space, ThingSpace)
            assert reloaded.space.group == "b"
    finally:
        space_mod._SPACES[:] = saved
        if os.path.exists(path):
            os.unlink(path)


def test_a_third_space_registers_ahead_of_the_catch_all():
    """Detection order is by fallback, not by arrival.

    ``SurfaceSpace`` claims any node without a transform, which is how a file
    written before spaces existed is recognised. A space registered by a third
    party -- necessarily after this package has registered its own -- must still be
    consulted first, or the catch-all takes its nodes and ``cortex.load`` returns an
    empty Dataset, since it swallows the per-view failure.
    """
    from cortex.dataset import _space as space_mod

    assert SurfaceSpace.fallback and not VolumeSpace.fallback
    assert [c.__name__ for c in space_mod.registered_spaces()] == [
        "VolumeSpace",
        "SurfaceSpace",
    ]

    ThingSpace = _third_space()[0]
    saved = list(space_mod._SPACES)
    try:
        space_mod.register_space(ThingSpace)
        assert [c.__name__ for c in space_mod.registered_spaces()] == [
            "VolumeSpace",
            "ThingSpace",
            "SurfaceSpace",
        ]
    finally:
        space_mod._SPACES[:] = saved


def test_a_third_space_renders_without_touching_quickflat():
    """``quickflat`` asks ``space.xfmname`` and ``view.dense`` and nothing else.

    It used to fork on ``hasattr(braindata, "xfmname")``, with a further
    ``isinstance(..., Volume2D/Vertex2D)`` inside each arm, so a third space took
    the surface arm and then failed. Uses this subject's real flatmap cache, so
    only the dispatch is synthetic.
    """
    from unittest import mock

    from cortex.quickflat import utils as qf

    _, Thing, _, _, n = _third_space()
    view = Thing(np.random.randn(n), subj)

    real_mask, real_extents = qf.get_flatmask(subj, height=64)
    npix = int(real_mask.sum())
    # A real sparse pixel-to-value matrix, just a synthetic one: every pixel takes
    # the first value. The renderer's arithmetic is genuinely exercised; only the
    # geometry it would have looked up is stood in for.
    from scipy import sparse

    pixmap = sparse.csr_matrix(
        (np.ones(npix), (np.arange(npix), np.zeros(npix, dtype=int))), shape=(npix, n)
    )

    with mock.patch.object(qf, "get_flatcache", return_value=pixmap) as cache:
        image, extents = qf.make_flatmap_image(view, height=64)

    # transposed relative to the mask, as the renderer returns it for display
    assert image.shape == real_mask.shape[::-1]
    # the space said "no transform", and that is what the cache was keyed on
    assert cache.call_args[0][1] is None


def test_a_space_with_no_webgl_encoding_says_so_once():
    """Only two wire encodings exist, and the space is where that is stated.

    ``dataset.js`` selects between them by testing ``mosaic === undefined``, so the
    limit is real; routing through the space does not remove it. What changes is
    that a space with no browser representation declines the capability with one
    message, instead of a consumer's ``else`` branch reaching ``volume.mosaic`` and
    dying on "Invalid data shape" several frames deep.
    """
    from cortex.webgl.data import Package

    _, Thing, _, _, n = _third_space()
    view = Thing(np.random.randn(n), subj)

    with pytest.raises(TypeError, match="ThingSpace has no webgl wire encoding"):
        Package(dataset.Dataset(view=view))


def test_a_third_space_reaches_the_browser_by_picking_an_encoding():
    """And with one of the two, it packages with no edit to ``webgl/data.py``."""
    from cortex.dataset._webgl import VertexAttributes
    from cortex.webgl.data import Package

    ThingSpace, Thing, _, _, n = _third_space()
    ThingSpace.pack_for_webgl = lambda self, data, *, raw: VertexAttributes(data, raw=raw)

    view = Thing(np.random.randn(n), subj)
    pkg = Package(dataset.Dataset(view=view))
    record = pkg.brains[view.name]

    assert pkg.images[view.name][0].dtype == np.float32
    assert record["raw"] is False
    assert "mosaic" not in record  # the JS reads this as "per-vertex attributes"
