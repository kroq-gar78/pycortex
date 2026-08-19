"""Characterization tests for the six public view classes.

Step 0 of DATASET_REFACTOR.md: written *before* the restructure, and asserting
what the classes do today rather than what they ought to do, so that the steps
after it can be judged behaviour-preserving. Every case is parametrized over the
full 2x3 grid -- two spaces (volumetric, surface) crossed with three channel
layouts (scalar, 2D, RGB) -- against each of the four things a view has to
survive: an HDF round trip, ``to_json``, ``.raw``, and ``quickflat``.

These are deliberately broad and shallow. They are not a substitute for the
targeted tests in ``test_dataset.py``; they are a net under the refactor, and the
thing they are most useful for is the case nobody thought to write down.
"""

import os
import tempfile

import cortex
import numpy as np
import pytest

from cortex import dataset

subj, xfmname, nverts, volshape = "S1", "fullhead", 304380, (31, 100, 100)


def _vol(scale=1.0):
    return np.random.randn(*volshape) * scale


def _vtx(scale=1.0):
    return np.random.randn(nverts) * scale


def _volume():
    return cortex.Volume(_vol(), subj, xfmname, cmap="hot", vmin=-2, vmax=2,
                         description="a volume")


def _vertex():
    return cortex.Vertex(_vtx(), subj, cmap="hot", vmin=-2, vmax=2,
                         description="a vertex")


def _volume2d():
    return cortex.Volume2D(_vol(), _vol(2), subj, xfmname, description="a volume2d")


def _vertex2d():
    return cortex.Vertex2D(_vtx(), _vtx(2), subj, description="a vertex2d")


def _volumergb():
    return cortex.VolumeRGB(_vol(), _vol(2), _vol(3), subj, xfmname,
                            description="a volumergb")


def _vertexrgb():
    return cortex.VertexRGB(_vtx(), _vtx(2), _vtx(3), subj, description="a vertexrgb")


#: name -> (factory, is_volumetric, channel layout). The grid, once, so every test
#: below covers all six rather than whichever two the author had in mind.
GRID = {
    "Volume": (_volume, True, "scalar"),
    "Vertex": (_vertex, False, "scalar"),
    "Volume2D": (_volume2d, True, "2d"),
    "Vertex2D": (_vertex2d, False, "2d"),
    "VolumeRGB": (_volumergb, True, "rgb"),
    "VertexRGB": (_vertexrgb, False, "rgb"),
}

ALL = sorted(GRID)
SCALAR = [k for k, v in GRID.items() if v[2] == "scalar"]
COLORMAPPED = sorted(k for k, v in GRID.items() if v[2] in ("scalar", "2d"))


# Five defects this file found on its first run, before anything was refactored.
# All five are now **fixed**, by the restructure rather than by being patched, and
# the marks that recorded them are gone -- which is what a strict xfail is for: it
# fails when the defect stops reproducing, so the fix cannot pass unnoticed. Kept
# here as a record of what each one was and what fixed it:
#
# - `Vertex2D` could not be reloaded at all. Slot 7 is null for a surface view, so
#   `xfmname` arrived as None rather than a per-channel list and `from_hdf` indexed
#   it unconditionally; `Dataset.from_file` swallows per-view exceptions, so the
#   view was silently *dropped*. Fixed by `space.view_xfmname`, which is None for a
#   space with no transform, and by `_from_hdf_view` no longer assuming a list.
# - An RGB view built without an explicit alpha reloaded **fully transparent**. The
#   synthesized all-ones alpha was written as a fourth data node, because `uniques`
#   and `_write_hdf` tested the `alpha` *property*, which builds one on demand and
#   is therefore never None. Reloaded, its bounds were percentiles of a constant
#   array -- vmin == vmax == 1 -- so normalizing divided by zero. Fixed by testing
#   `_alpha`, the stored field.
# - `VertexRGB` movies were unconstructible: the default alpha was sized from the
#   sampled array's vertex count alone, which carries no frame axis. Fixed by
#   sizing from the channel's stored array, which has the frame axis exactly when
#   the channels do.
# - `Volume2D` had no `.volume`, though `Vertex2D` had `.vertices`. Fixed by
#   `Dataview2D.dense`, one accessor on the column that both now forward to.
# - `description` reloaded as `bytes` for all six. Fixed by decoding slot 1 in
#   `from_hdf`.


@pytest.fixture
def hdf_path():
    path = tempfile.mktemp(suffix=".hdf")
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_the_view_knows_its_own_subject_and_space(name):
    factory, volumetric, _ = GRID[name]
    view = factory()

    assert view.subject == subj
    assert isinstance(view, getattr(cortex, name))
    # `xfmname` is the one piece of geometry every consumer reaches for, and only
    # the volumetric half has it -- which is exactly why consumers duck-type it.
    assert hasattr(view, "xfmname") is volumetric
    if volumetric:
        assert view.xfmname == xfmname


@pytest.mark.parametrize("name", COLORMAPPED)
def test_a_colormapped_view_has_a_colormap_and_bounds(name):
    view = GRID[name][0]()
    assert view.cmap is not None
    cmapdict = view.get_cmapdict()
    assert set(cmapdict) >= {"cmap", "vmin", "vmax"}


# ----------------------------------------------------------------------
# the sampled array
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_the_sampled_array_has_a_leading_frame_axis(name):
    factory, volumetric, layout = GRID[name]
    view = factory()

    sampled = view.volume if volumetric else view.vertices
    spatial = volshape if volumetric else (nverts,)
    if layout == "scalar":
        assert sampled.shape == (1,) + spatial
    else:
        # 2D and RGB views are colormapped before they are sampled, so what comes
        # back is uint8 RGBA rather than the stored scalars.
        assert sampled.shape == (1,) + spatial + (4,)
        assert sampled.dtype == np.uint8


# ----------------------------------------------------------------------
# .raw
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_raw_is_an_rgb_view_of_the_same_space(name):
    factory, volumetric, _ = GRID[name]
    raw = factory().raw

    assert isinstance(raw, cortex.VolumeRGB if volumetric else cortex.VertexRGB)
    assert raw.subject == subj
    sampled = raw.volume if volumetric else raw.vertices
    assert sampled.dtype == np.uint8
    assert sampled.shape[-1] == 4


# ----------------------------------------------------------------------
# to_json
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_to_json_describes_the_view_and_its_data(name):
    factory, volumetric, layout = GRID[name]
    view = factory()

    full = view.to_json(simple=False)
    assert full["desc"] == view.description
    assert "attrs" in full and "state" in full
    if layout != "rgb":
        # RGB views carry their own colours, so they have no colormap to describe.
        assert full["cmap"] and full["vmin"] and full["vmax"]

    # `uniques` yields the data-carrying views inside this one, and each of those
    # is what the browser is actually sent.
    for packable in view.uniques(collapse=True):
        simple = packable.to_json(simple=True)
        assert simple["subject"] == subj
        assert simple["name"] == packable.name
        assert simple["min"] <= simple["max"]
        # how to unpack the array: a 3-D grid for volumes, a split point for surfaces
        assert ("shape" in simple) is volumetric
        assert ("split" in simple) is not volumetric


@pytest.mark.parametrize("name", ALL)
def test_uniques_collapses_to_content_addressed_names(name):
    view = GRID[name][0]()
    names = [p.name for p in view.uniques(collapse=True)]

    assert names and all(n.startswith("__") for n in names)
    # A set, not a list: two views over identical data must collapse to one entry,
    # since the name is a content hash and is what the data is stored under.
    assert len(set(names)) == len(names)


# ----------------------------------------------------------------------
# HDF round trip
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_the_view_survives_an_hdf_round_trip(name, hdf_path):
    factory, volumetric, layout = GRID[name]
    view = factory()

    dataset.Dataset(**{name: view}).save(hdf_path)
    reloaded = cortex.load(hdf_path)[name]

    assert type(reloaded) is type(view)
    assert reloaded.subject == view.subject
    assert reloaded.description == view.description
    if volumetric:
        assert reloaded.xfmname == view.xfmname
    if layout != "rgb":
        assert reloaded.cmap == view.cmap
        assert np.allclose(float(reloaded.vmin), float(view.vmin))
        assert np.allclose(float(reloaded.vmax), float(view.vmax))

    # and the values themselves, through whatever accessor the layout exposes
    before = view.volume if volumetric else view.vertices
    after = reloaded.volume if volumetric else reloaded.vertices
    assert np.allclose(np.nan_to_num(before), np.nan_to_num(after))


@pytest.mark.parametrize("name", ALL)
def test_the_description_survives_an_hdf_round_trip(name, hdf_path):
    """Separate from the round trip above so one defect does not mask the rest."""
    view = GRID[name][0]()
    dataset.Dataset(v=view).save(hdf_path)

    reloaded = cortex.load(hdf_path)["v"]
    assert reloaded.description == view.description


@pytest.mark.parametrize("name", ALL)
def test_a_movie_survives_an_hdf_round_trip(name, hdf_path):
    _, volumetric, layout = GRID[name]
    frames = 3

    def movie(scale=1.0):
        shape = (frames,) + (volshape if volumetric else (nverts,))
        return np.random.randn(*shape) * scale

    cls = getattr(cortex, name)
    spec = (subj, xfmname) if volumetric else (subj,)
    if layout == "scalar":
        view = cls(movie(), *spec)
    elif layout == "2d":
        view = cls(movie(), movie(2), *spec)
    else:
        view = cls(movie(), movie(2), movie(3), *spec)

    sampled = view.volume if volumetric else view.vertices
    assert sampled.shape[0] == frames

    dataset.Dataset(m=view).save(hdf_path)
    reloaded = cortex.load(hdf_path)["m"]
    assert type(reloaded) is type(view)
    after = reloaded.volume if volumetric else reloaded.vertices
    assert after.shape == sampled.shape


# ----------------------------------------------------------------------
# quickflat
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_quickflat_renders_the_view(name):
    from cortex.quickflat.utils import make_flatmap_image

    factory, _, layout = GRID[name]
    image, extents = make_flatmap_image(factory(), height=256)

    assert len(extents) == 4
    # A colormapped view renders to one scalar per pixel for imshow to map; a
    # pre-coloured one renders straight to RGBA.
    assert image.ndim == (2 if layout == "scalar" else 3)
    if image.ndim == 3:
        assert image.shape[-1] == 4


# ----------------------------------------------------------------------
# the webgl wire format
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_the_view_packages_for_webgl(name):
    from cortex.webgl.data import Package

    factory, volumetric, layout = GRID[name]
    view = factory()
    pkg = Package(dataset.Dataset(view=view))

    # 2D views decompose into their two scalar channels on the way out, so the
    # number of shipped arrays follows the layout rather than the class.
    expected = {"scalar": 1, "2d": 2, "rgb": 1}[layout]
    assert len(pkg.images) == expected
    assert len(pkg.brains) == expected

    for record in pkg.brains.values():
        # `mosaic` present means "unpack this PNG as a 3-D texture"; absent means
        # "these are per-vertex attributes". This is the one dispatch the browser
        # makes, so it is the one that must not drift.
        assert ("mosaic" in record) is volumetric
        assert record["raw"] is (layout == "rgb")

    metadata = pkg.metadata()
    assert set(metadata) == {"views", "data", "images"}
    assert len(metadata["views"]) == 1
