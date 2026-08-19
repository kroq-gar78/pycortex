# `cortex.dataset` as three composed layers

What was built on this branch, and where it departs from
[DATASET_REFACTOR.md](../../DATASET_REFACTOR.md), which is the design it
implements. Read that first for the diagnosis; this is the record of the result.

## The shape

```
Dataview          how values become color -- cmap, vmin/vmax, description, state, priority
   holds BrainData      the values -- the array, movie-ness, the content hash, the operators
      holds Space       the geometry -- subject, and whatever else locates one value
```

Composition only. No class in the graph inherits both a view and its data, which
is what `Volume(VolumeData, Dataview)` did, and what made `BrainData.to_json` and
`VolumeData.copy` call `super()` methods that existed nowhere in their own
ancestry, resolving only because a subclass's MRO happened to thread through the
other base.

| | file | what it owns |
| --- | --- | --- |
| `Space`, `VolumeSpace`, `SurfaceSpace` | `_space.py` | geometry, and the only `db` access |
| `BrainData` | `braindata.py` | one array plus its space |
| `Dataview`, `DataviewScalar`, `Volume`, `Vertex` | `views.py` | the root, and the scalar column |
| `Dataview2D`, `Volume2D`, `Vertex2D` | `view2D.py` | the 2D column |
| `DataviewRGB`, `VolumeRGB`, `VertexRGB` | `viewRGB.py` | the RGB column |
| `WebGLPayload`, `MosaicTexture`, `VertexAttributes` | `_webgl.py` | the two wire encodings |
| `_hash`, `_hdf_write` | `_hdf.py` | serialization helpers |

The six public classes keep their names, their signatures, and their identity for
`isinstance`. They are thin: build a space, build a `BrainData`, hand both up.

## Spaces are values

Immutable, hashable, comparable. This is what `space1 == space2` replaced:
`Volume2D.raw`, `VolumeRGB.__init__` and `Vertex2D.__init__` each spelled out
their own version of "are these two the same place", and none of the three agreed
on what to compare.

It also makes sharing safe, which is the other half of the payoff. `db.get_surf`
ran for **every** `Vertex` constructed, to learn two integers that depend on
nothing but the subject; those lookups are now cached on what they depend on, and
`copy()` carries the space over rather than rebuilding it, so nothing re-resolves
a mask.

### `resolve_for`: the one part that is not a pure move

A flat volume's mask and a half-length vertex array's hemisphere are facts about
*that array*, so they cannot live on a value shared between views.
`space.resolve_for(data)` returns the space the array is really in — a new value,
never a mutation — and `BrainData` holds the result:

```python
self.space = space.resolve_for(data)     # which mask does this flat array match?
self._data = self.space.validate(data)   # and is it the right size for it?
```

That split is exactly where `VolumeData._check_size` mixed data-derived facts
(`linear`, `movie`) with space-derived ones (`shape`, `mask`) in a single pass.
DATASET_REFACTOR.md §4.2 called the untangling the highest-risk part of the plan,
and it is: everything else in the extraction is a move.

One casualty. `VertexData.hem` — "left", "right" or "both" — described the array
*before* padding, and padding happens inside `validate`. It is a fact about an
array that no longer exists by the time anyone could ask, so it is dropped rather
than faked. `SurfaceSpace.hemisphere_of(data)` answers the same question for a
caller who still has the unpadded array. Nothing in this repository used it.

## What a new kind of brain data costs

One `Space` subclass, plus three view classes that are a constructor each. The
three columns supply colormapping, `.raw`, `uniques`, `to_json`, HDF, the
arithmetic operators, NaN handling and alpha. `test_space.py` builds a
`ThingSpace` over ten unrelated values — neither volumetric nor surface, so
nothing it does can accidentally take a built-in path — and exercises all of it,
including an HDF round trip that needs no factory edits.

Ten members are abstract, plus `spec_keys` if the space takes any argument
besides `subject`:

```python
class MySpace(Space):
    spec_keys = ("group",)             # its constructor args besides `subject`
    def _identity(self): ...           # what makes two of these the same space
    @property
    def xfmname(self): ...             # the transform to sample through, or None
    @property
    def shape(self): ...               # one frame of dense data
    def resolve_for(self, data): ...   # the space this array is really in
    def validate(self, data): ...      # check it, and return it in stored form
    def is_movie(self, data): ...      # leading time axis?
    def to_dense(self, data): ...      # what `.volume` / `.vertices` publish
    def to_json(self): ...             # space keys for the full JSON
    def write_hdf_attrs(self, h5, node): ...
    @classmethod
    def from_hdf(cls, attrs, *, subject, xfmname, mask): ...   # or None, if not ours
```

and four more are concrete, to be overridden only when they are wrong:
`template_shape` (what `empty`/`random` size themselves from, defaulting to
`shape`), `describe_layout(data)` (the keys telling the browser how to unpack an
array, which is per-array rather than per-space and so takes the data),
`align(a, b)` (two arrays in a layout where position *i* means the same place in
both — what a 2D view needs, and what `VolumeSpace` overrides because a flat
array's positions mean something only relative to its own mask), and
`pack_for_webgl`.

The space also names the three classes that view its data —
`space.scalar_view`, `twod_view`, `rgb_view` — assigned at the bottom of
`views.py` once all three columns exist. That is what `.raw` and the HDF factories
dispatch through, so neither names a concrete class. It replaces the `_cls` class
pointer, which existed to call unbound methods of whichever data class a subclass
had in mind: `self._cls._write_hdf(self.red, h5)`.

## Consumers dispatch on the space, never on the view class

The rule that collapses an N×M matrix to N+M. Before, each consumer enumerated
the grid by hand, differently:

| site | was | now |
| --- | --- | --- |
| `webgl/data.py` | `isinstance(brain, (Vertex, VertexRGB))`, three times | `space.pack_for_webgl(brain.dense, raw=...)` |
| `quickflat/utils.py` | `hasattr(braindata, "xfmname")`, with an `isinstance` inside each arm | `space.xfmname` and `view.dense`, no branch |
| `export/save_views.py` | `isinstance(volume, (Volume, Volume2D, VolumeRGB))` | `isinstance(volume.space, VolumeSpace)` |
| `blender/__init__.py` | `isinstance(braindata, dataset.braindata.VertexData)` | `isinstance(braindata.space, SurfaceSpace)` |
| `dataset.py` (packing) | `isinstance(data, Volume)` then `data._mask` | `isinstance(space, VolumeSpace)` then `space.mask_spec` |

`quickflat` is the clean case: it asks the two questions it actually has —
what to sample through, and what to sample — and has no branch left at all.

Three consumers are still deliberately narrow, because they need a capability
rather than a kind: `volume.show_slice` needs a `Volume` to slice against an
anatomical reference, `blender.add_cutdata` needs a single `cmap`, and
`Mapper.__call__` splits per-hemisphere by vertex count. Those correctly reject a
third space rather than mis-drawing it.

### webgl is the one axis that is not fully open

`dataset.js` selects its path by testing `mosaic === undefined`, so exactly two
wire encodings exist, and routing through the space does not change that — it only
moves where it is said. `Space.pack_for_webgl` is concrete and raises by default,
naming both encodings and the JS file; a space picks `MosaicTexture` or
`VertexAttributes`, or adds a third branch to the JS. A space with no browser
representation at all is a legitimate thing to have, since `quickflat` needs only
`to_dense`, so declining is not a failure to implement something.

Both encodings live in `_webgl.py` as a compatibility surface. Four consequences
follow from one decision — the dtype cast, whether alpha is premultiplied, whether
frames are mosaicked into PNGs or shipped as per-vertex attributes, and whether
they are permuted into the CTM's vertex order — and `webgl/data.py` used to fork
on the same fact once per consequence. The premultiplied-alpha asymmetry is a
fact about Three.js (`tex.premultiplyAlpha` on texture upload, nothing at all for
vertex attributes) and is pinned in both directions by `test_webgl_data.py`.

## Deviations from DATASET_REFACTOR.md

Four, each with its reason.

**1. Composite views hold scalar views, not `BrainData`.** §3.3 sketches
`Dataview2D` as holding two `BrainData` and `DataviewRGB` as holding four. They
hold `DataviewScalar`s instead, because `.dim1` and `.red` are public and are
scalar views today, and because a channel's `vmin`/`vmax` have to live somewhere:
`Dataview2D` defaults each axis to its own channel's range, and the RGB column
normalizes each channel by its own bounds. A `BrainData` carries no bounds. The
composition is still uniform and the diamond is still gone — a channel is a
`Dataview` over a `BrainData` — but the layering is four deep for a composite
rather than three.

**2. `isinstance(x, VolumeData)` does not survive.** §4.1 promises the public
surface does not move, and it does not: all six classes keep their names and their
`isinstance` behaviour. But `VolumeData`/`VertexData` are non-public names, they
are now aliases of the single `BrainData`, and a view *holds* a `BrainData` rather
than being one — so `isinstance(vol, VolumeData)`, which used to be true for a
`Volume`, is false. This is unavoidable under the proposed layering rather than a
shortcut: it is what "composed, not inherited" means. Ask about the space.

**3. `sample_flatmap` and `to_dense` are one method, not two.** §3.1 lists both.
`quickflat` turned out to need only the array and the transform name, and the
flatmap machinery (`get_flatcache`, the pixel-weight dot product) is the
renderer's, not the geometry's. Adding a `sample_flatmap` that only forwarded
`to_dense` would put a second name on one thing.

**4. Unknown constructor keywords are still absorbed into `attrs`.** §4.2 step 2
says to reject them. `attrs` is a genuine feature — `stim` is read by
`webgl/view.py`, `alpha` by the 2D column's `raw`, and users hang their own
metadata there — so rejecting *unknown* keywords would remove it, and rejecting
only misspellings is a separate piece of work from this restructure. Left as is,
so `Volume(data, subj, xfm, cmpa="hot")` remains a silently ignored attribute.

## Deletions

Per §4.4, done early because they shrink the surface being refactored:

- `Multiview`, whose `__init__` raised `NotImplementedError` on its third line.
- `Dataview.from_hdf`'s multi-view branch, which built a list of views and then
  raised unconditionally. The `len(data) != 1` guard now says so directly.
- `blend_curvature`, already deprecated, and wired into three classes by hand —
  once as a method, once as `blend_curvature = _cls.blend_curvature  # hacky
  inheritance`. Its test is deleted with it. The replacement it already
  recommended in its own deprecation warning is `Vertex2D`/`Volume2D` with an
  alpha-encoding 2D colormap.

## Behaviour, and the five defects the net caught

`test_characterization.py` was written **before** any restructuring (§4.2 step 0):
the full 2×3 grid against each of HDF save/load, `to_json`, `.raw` and
`quickflat`. It found five live defects on its first run, which it recorded as
strict xfails so that a step which repaired one could not do so unnoticed. All
five are fixed, and by the restructure rather than by being patched:

| defect | what fixed it |
| --- | --- |
| `Vertex2D` silently *dropped* on reload — slot 7 is null for a surface view, `from_hdf` indexed it unconditionally, and `Dataset.from_file` swallows per-view exceptions | `space.view_xfmname` is None for a space with no transform |
| a default-alpha RGB view reloaded **fully transparent** — the synthesized all-ones alpha was written as a fourth data node, because `uniques` and `_write_hdf` tested the `alpha` *property*, which builds one on demand and is never None; reloaded, its bounds were percentiles of a constant array, so normalizing divided by zero | testing `_alpha`, the stored field |
| `VertexRGB` movies unconstructible — the default alpha was sized from the sampled array's vertex count, which has no frame axis | sizing from the channel's stored array |
| `Volume2D` had no `.volume`, though `Vertex2D` had `.vertices` | `Dataview2D.dense`, one accessor on the column |
| `description` reloaded as `bytes`, for all six | decoding slot 1 in `from_hdf` |

A sixth, found while diffing rather than by a test: a float32 view could not be
saved at all, because its defaulted `vmin` is an `np.float32` and `json.dumps`
refuses one. `views._dumps` is the single JSON boundary that converts numpy
scalars, and the conversion belongs there rather than in
`_resolve_percentiles` — the numpy scalar has to survive on the view, since under
NEP 50 it is a *strong* operand, and demoting it to a Python float changes a
channel's last bit and therefore its content hash and its on-disk node name.

Everything else is byte-identical to `types-orig`: the eight HDF view slots, the
`to_json` output in both modes, every content-addressed data node name, and
everything `Package` ships — checked across the grid plus movies and RGB-with-alpha
on both sides of `reorder`.

## The wire format is still a hard interface

`webgl/resources/js/dataset.js` dispatches structurally; no Python class name
reaches the browser. §4.3 keeps the JS side out of this pass, so these shapes are
fixed:

| JS test | meaning |
| --- | --- |
| `mosaic === undefined` | surface-style per-vertex data |
| `json.data[i] instanceof Array` | 2D view |
| `json.raw` | RGB, 4-channel uint8 texture path |
| `json.vmin[0] instanceof Array` | 2D ranges |

Slot layout, written by `Dataview._write_view_node`:

| slot | contents |
| --- | --- |
| 0 | data node name(s); a nested list means a 2D or RGB view |
| 1 | description |
| 2 | `[cmap]`, or `"null"` for RGB |
| 3 | `[vmin]`, or `[[vmin, vmin2]]` for 2D, or `"null"` for RGB |
| 4 | `[vmax]`, likewise |
| 5 | state |
| 6 | attrs |
| 7 | `space.view_xfmname` — `[xfmname]` for volumes, `null` for surfaces |

An RGB view built with `alpha=None` now writes `null` in the alpha position of
slot 0 rather than a synthesized channel, which is both the fix above and what
makes `_from_hdf_view`'s `data[3] is not None` branch reachable.

## Not done: step 6, electrodes

`ElectrodeSpace` is the point of the exercise and is deliberately absent. §5 leaves
two questions open that change its serialization, and both have to be settled
first: whether montages live in the filestore (so several datasets share one and
coordinates are not duplicated into every HDF file) or travel with the data
object, and what coordinate space the coordinates are in. The JS renderer for
point geometry is separate follow-on work either way, per §4.3.

One thing this branch does settle, against §5's third question: cross-space
mapping is *not* a `Space` method. `Volume.map`, `Vertex.map` and `Vertex.volume`
transform between spaces via `cortex.utils.get_mapper`, so they are a property of a
space *pair*, and putting them on `Space` would drag the mapper into `_space.py`.
An electrode-to-cortex mapper belongs beside the existing ones.
