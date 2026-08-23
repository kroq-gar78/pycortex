"""Tests for webviewer export helpers that require the headless viewer.

These tests require ``playwright`` and Chromium to be installed::

    pip install playwright
    playwright install chromium
"""
import os
import tempfile

import numpy as np
import pytest

import cortex

from .testing_utils import has_playwright, has_playwright_firefox

pytestmark = pytest.mark.skipif(
    not has_playwright,
    reason="playwright + Chromium not available",
)


subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)


def test_save_3d_views_headless():
    """save_3d_views with headless=True should produce an image file."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test_img")
        file_names = cortex.export.save_3d_views(
            vol,
            base_name=base,
            list_angles=["lateral_pivot"],
            list_surfaces=["inflated"],
            size=(1024, 768),
            trim=False,
            # The WebGL scene needs time to initialise surfaces before
            # _set_view can succeed; sleep=10 (the default) is safe.
            sleep=10,
            headless=True,
        )

        assert len(file_names) == 1
        assert os.path.isfile(file_names[0])
        assert os.path.getsize(file_names[0]) > 0

        # Check that the file is a valid image and has the expected dimensions.
        from PIL import Image
        with Image.open(file_names[0]) as img:
            assert img.size == (1024, 768)


def test_plot_panels_headless():
    """plot_panels with headless=True should produce an output image file."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    panels = [
        cortex.export.PanelParams({
            "extent": (0.0, 0.0, 1.0, 1.0),
            "view": cortex.export.PanelView(angle="lateral_pivot", surface="inflated"),
        })
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_name = os.path.join(tmpdir, "panels.png")
        fig = cortex.export.plot_panels(
            vol,
            panels=panels,
            figsize=(8, 6),
            windowsize=(1024, 768),
            save_name=save_name,
            sleep=10,
            viewer_params={},
            headless=True,
        )

        # The function returns a matplotlib Figure and, when save_name is
        # provided, should have written the file to disk.
        assert fig is not None
        assert os.path.isfile(save_name)
        assert os.path.getsize(save_name) > 0


def _vertex_dataviews():
    """Build one dataview per vertex-backed type, on a fixed random seed."""
    npts = cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]
    rng = np.random.RandomState(0)
    first = np.sin(np.linspace(0, 40, npts)) + 0.3 * rng.randn(npts)
    second = np.cos(np.linspace(0, 17, npts))

    return {
        "Vertex": cortex.Vertex(first, subj, vmin=-2, vmax=2),
        "Vertex2D": cortex.Vertex2D(
            first, second, subj, vmin=-2, vmax=2, vmin2=-1, vmax2=1
        ),
        "VertexRGB": cortex.VertexRGB(
            np.clip((first + 2) * 60, 0, 255),
            np.clip((second + 1) * 120, 0, 255),
            np.full(npts, 80.0),
            subj,
        ),
    }


def _render_headless(dv, tmpdir, name, browser):
    from PIL import Image

    file_names = cortex.export.save_3d_views(
        dv,
        base_name=os.path.join(tmpdir, f"{name}_{browser}"),
        list_angles=["lateral_pivot"],
        list_surfaces=["inflated"],
        size=(600, 450),
        trim=False,
        sleep=10,
        headless=True,
        browser=browser,
    )
    assert len(file_names) == 1
    assert os.path.isfile(file_names[0])
    with Image.open(file_names[0]) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64)


@pytest.mark.parametrize("browser", ["chromium", "firefox"])
@pytest.mark.parametrize("kind", ["Vertex", "Vertex2D", "VertexRGB"])
def test_vertex_dataviews_render_non_blank(kind, browser):
    """Vertex-backed dataviews must actually render.

    The surface_vertex shader sits close to the WebGL minimum of 16 vertex
    attributes.  When it declares more than that, the program silently fails to
    link and the surface renders as a blank image with no Python-side error --
    this regressed Vertex2D in *both* browsers and Vertex/VertexRGB under
    Firefox's Mesa software rasterizer, which (unlike Chromium's ANGLE) counts
    declared-but-unused attributes against the limit.
    """
    if browser == "firefox" and not has_playwright_firefox:
        pytest.skip("playwright + Firefox not available")

    dv = _vertex_dataviews()[kind]
    with tempfile.TemporaryDirectory() as tmpdir:
        image = _render_headless(dv, tmpdir, kind, browser)

    # A failed shader link yields a uniformly blank canvas.
    assert image.std() > 1.0, (
        f"{kind} rendered a blank image in {browser} (std={image.std():.4f}); "
        "the surface shader most likely failed to link."
    )


@pytest.mark.skipif(
    not (has_playwright and has_playwright_firefox),
    reason="playwright + both Chromium and Firefox are required",
)
@pytest.mark.parametrize("kind", ["Vertex", "Vertex2D", "VertexRGB"])
def test_vertex_dataviews_chromium_matches_firefox(kind):
    """Each vertex dataview type should look the same in both browsers."""
    dv = _vertex_dataviews()[kind]

    with tempfile.TemporaryDirectory() as tmpdir:
        images = {
            browser: _render_headless(dv, tmpdir, kind, browser)
            for browser in ("chromium", "firefox")
        }

    assert images["chromium"].shape == images["firefox"].shape
    mean_abs_diff = np.abs(images["chromium"] - images["firefox"]).mean()
    assert mean_abs_diff < 5.0, (
        f"{kind} differs between Chromium and Firefox (mean abs diff "
        f"= {mean_abs_diff:.2f} out of 255)."
    )


@pytest.mark.skipif(
    not (has_playwright and has_playwright_firefox),
    reason="playwright + both Chromium and Firefox are required",
)
def test_save_3d_views_headless_chromium_matches_firefox():
    """The same scene rendered headlessly in Chromium and Firefox should
    produce near-identical images, confirming Firefox's software WebGL
    rendering is not silently producing a blank/broken view."""
    from PIL import Image

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with tempfile.TemporaryDirectory() as tmpdir:
        images = {}
        for browser in ("chromium", "firefox"):
            base = os.path.join(tmpdir, f"test_img_{browser}")
            file_names = cortex.export.save_3d_views(
                vol,
                base_name=base,
                list_angles=["lateral_pivot"],
                list_surfaces=["inflated"],
                size=(1024, 768),
                trim=False,
                sleep=10,
                headless=True,
                browser=browser,
            )
            assert len(file_names) == 1
            assert os.path.isfile(file_names[0])
            with Image.open(file_names[0]) as img:
                images[browser] = np.asarray(img.convert("RGB"), dtype=np.float64)

        assert images["chromium"].shape == images["firefox"].shape

        # Software WebGL rasterizers can differ slightly in anti-aliasing
        # and float precision, so compare with a tolerance rather than
        # requiring byte-for-byte equality.
        diff = np.abs(images["chromium"] - images["firefox"])
        mean_abs_diff = diff.mean()
        assert mean_abs_diff < 5.0, (
            f"Chromium and Firefox renders differ too much (mean abs diff "
            f"= {mean_abs_diff:.2f} out of 255); Firefox rendering may be "
            "broken."
        )