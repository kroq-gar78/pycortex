# Refactoring `cortex.dataset`

A design proposal: untangle the `BrainData` / `Dataview` multiple-inheritance
structure, and make new data domains (starting with intracranial EEG) additions
rather than rewrites.

---

## 1. Diagnosis

The module conflates two orthogonal concerns:

- **Where the values live** — the geometry/domain. Volume (subject + xfm + mask),
  surface (subject + vertex counts per hemisphere), and in future electrodes.
- **How the values become colors** — the view. 1D colormap, 2D colormap, RGB(A)
  passthrough, plus `vmin`/`vmax`/`description`/`state`/`priority`.

The problem is not that the two are combined. It is that they are combined **two
different ways in the same module**:

| Class | Strategy |
|---|---|
| `Volume(VolumeData, Dataview)` | fused by multiple inheritance |
| `Vertex(VertexData, Dataview)` | fused by multiple inheritance |
| `Volume2D(Dataview2D)` holding `dim1, dim2: Volume` | composed |
| `VolumeRGB(DataviewRGB)` holding `red, green, blue: Volume` | composed |
| `Vertex2D`, `VertexRGB` | composed |

So the scalar case is an inheritance diamond and every other case is composition.
Everything below follows from that split.

### 1.1 The cooperative-`__init__` kwargs chain

```
Volume.__init__ → VolumeData.__init__ → BrainData.__init__ → super().__init__(**kwargs) → Dataview.__init__
```

`cmap`, `vmin`, `vmax` travel as opaque `**kwargs` through two classes that have
no idea what they are. Anything unrecognized lands silently in
`Dataview.attrs`, which is a `**kwargs` sink ([views.py:174](cortex/dataset/views.py:174)) — so
`Volume(data, subj, xfm, cmpa="hot")` is not an error, it is a silently ignored
attribute.

Meanwhile the RGB and 2D classes are *not* in that chain, so the same kwarg is a
`TypeError` there. `_from_hdf_data` has to hand-filter kwargs to work around
exactly this ([views.py:82-87](cortex/dataset/views.py:82) and again at
[views.py:141-144](cortex/dataset/views.py:141)):

```python
rgb_kwargs = {k: v for k, v in kwargs.items() if k in ("description", "state", "priority")}
```

One construction path per class, with different rules about what is accepted.

### 1.2 Neither base class is self-contained

- `BrainData.to_json` calls `super().to_json()` — which exists only on `Dataview`.
- `BrainData` has no `copy`, yet `VolumeData.copy` calls `super().copy(...)`,
  which resolves to `Dataview.copy`.
- `Dataview.raw` reads `self.data`, which exists only on `BrainData`. The code
  says so out loud ([views.py:328](cortex/dataset/views.py:328)):
  `# TODO: self.data relies on BrainData. Would need common inheritance for this to work.`

So `VolumeData` alone is broken, `Dataview` alone is broken, and each is a
half-object that only closes when the other is mixed in. Reading `VolumeData`
tells you very little about what `Volume` actually does.

### 1.3 Method origins are invisible at the call site

For a single `Volume`:

- `.uniques()` comes from `BrainData`
- `.copy()` comes from `VolumeData`, which delegates to `Dataview`
- `.to_json()` comes from `VolumeData` → `BrainData` → `Dataview` (three files, one call)
- `.raw` comes from `Volume`, which calls `Dataview.raw` via `super()`

This is precisely the "poor mental modeling of the origin of inherited values"
that has been producing bugs.

### 1.4 The composed classes duplicate `Dataview.__init__` instead of calling it

`Dataview2D.__init__` ([view2D.py:22-33](cortex/dataset/view2D.py:22)) and
`DataviewRGB.__init__` ([viewRGB.py:84-102](cortex/dataset/viewRGB.py:84)) both
re-set `state` / `attrs` / `priority` / `description` by hand. Neither calls
`Dataview.__init__`. The `priority` default of `1` is written out three times.
Add a field to `Dataview` and you must remember three other places.

### 1.5 Manual dispatch through a `_cls` class attribute

Because the view classes cannot call the data classes polymorphically, they call
unbound methods through a class pointer:

```python
self._cls._write_hdf(self.red, h5)          # viewRGB.py:129
blend_curvature = _cls.blend_curvature      # view2D.py:244 — comment: "hacky inheritance"
```

### 1.6 There is no single way to ask "what space is this?"

Since `Volume2D` and `VolumeRGB` do not inherit `VolumeData`, no single
`isinstance` answers the question. Consumers each invented their own test:

| Site | Test |
|---|---|
| [webgl/data.py:35,39,54,69,94](cortex/webgl/data.py:35) | `isinstance(brain, (dataset.Vertex, dataset.VertexRGB))` — five separate branches |
| [quickflat/utils.py:44](cortex/quickflat/utils.py:44) | `hasattr(braindata, "xfmname")` |
| [export/save_views.py:125](cortex/export/save_views.py:125) | `isinstance(volume, (cortex.Volume, cortex.Volume2D, cortex.VolumeRGB))` |
| [blender/__init__.py:174](cortex/blender/__init__.py:174) | `isinstance(braindata, dataset.braindata.VertexData)` |
| [volume.py:144](cortex/volume.py:144) | `isinstance(dataview, dataset.Volume)` |

This is an N×M matrix — *spaces* × *view types* — enumerated by hand at every
consumer. It is the single biggest reason a new domain is a rewrite rather than
an addition.

### 1.7 Temporal coupling in `__init__`

`DataviewRGB.__init__` reads `self.red.subject` before it has been passed
anything, so subclasses must assign `self.red` *before* calling `super().__init__`.
`VolumeRGB.__init__` then reads `self.alpha.xfmname`
([viewRGB.py:548](cortex/dataset/viewRGB.py:548)) through a property whose getter
*constructs a new `Volume`* from `self.red`. Half-initialized objects calling
properties that build other objects is the shape of #612 / #632.

### 1.8 `vmin`/`vmax` defaulting happens in three places, three ways

- `Volume.__init__` / `Vertex.__init__`: compute percentiles eagerly and **store** them.
- `Dataview.to_json`: compute percentiles lazily **if still None**.
- `Dataview2D.__init__`: pull them from `dim1` / `dim2`.

So whether `view.vmin is None` after construction depends on which class you
built. Any consumer reading `.vmin` has to know which.

### 1.9 Dead and deprecated code still in the graph

- `Multiview.__init__` raises `NotImplementedError` on its third line ([views.py:343](cortex/dataset/views.py:343)).
- `Dataview.from_hdf`'s multi-view branch raises `NotImplementedError` ([views.py:270](cortex/dataset/views.py:270)) after doing the work.
- `blend_curvature` is already deprecated, but is still wired into three classes by hand.

---

## 2. What iEEG needs, and why it does not fit today

An electrode dataset is: **N points with (x, y, z) coordinates**, per-electrode
metadata (name like `STG6`, parent device/strip/grid, shape/size, possibly a
good/bad flag), and values of shape `(N,)` or `(t, N)`.

Structurally it differs from the existing domains in three ways:

1. **The geometry travels with the data.** `VolumeData` gets its shape from
   `db.get_xfm`; `VertexData` gets its lengths from `db.get_surf`. Both derive
   geometry from the database given `(subject, xfmname)`. There is no equivalent
   for electrodes unless we decide the filestore stores montages (see open
   question 4) — the coordinates are part of the object.
2. **It carries per-element metadata that must survive everything.** Names and
   device grouping have to survive `copy()`, HDF round-trip, and the JSON sent to
   the viewer. There is no place for that today: every `copy()` signature is
   hand-written per class (`VolumeData.copy` passes `subject`, `xfmname`, `mask`),
   and HDF metadata is ad-hoc string attrs on the data node.
3. **It has no flatmap projection and no per-vertex/voxel texture.** `quickflat`
   can only draw it as markers; WebGL needs a third renderer (instanced spheres
   or point sprites) alongside the existing texture path (`VolumeData`) and vertex-attribute
   path (`VertexData`) in [dataset.js](cortex/webgl/resources/js/dataset.js:318).

But it wants **all** the existing view machinery: 1D colormaps, 2D colormaps,
RGB, vmin/vmax, movies, `Dataset`, HDF save, `.raw`.

**Under today's structure, adding it means writing four classes** —
`ElectrodeData(BrainData)`, `Electrodes(ElectrodeData, Dataview)`,
`Electrodes2D(Dataview2D)`, `ElectrodesRGB(DataviewRGB)` — each replicating the
same patterns, **plus editing every dispatch site in §1.6**. That is the cost we
should refuse to pay, and it will recur for the next domain after this one.

---

## 3. Proposal: Space / BrainData / Dataview, composed uniformly

Three layers, each with one job, related by composition only.

```
Dataview            how values become colors     (cmap, vmin/vmax, description, state, priority)
   └── holds BrainData
                    values                       (ndarray + movie flag + hashing + numpy ops)
          └── holds Space
                    geometry                     (subject, and whatever else locates a value)
```

### 3.1 `Space` — the geometry, no values

```python
class VolumeSpace(Space):     # subject, xfmname, mask
class VertexSpace(Space):     # subject  (llen/rlen from db)
class ElectrodeSpace(Space):  # subject, coords (N,3), names, devices, shapes, coord_space
```

Properties:

- **Value-like**: immutable, hashable, `__eq__`. `space1 == space2` replaces the
  ad-hoc subject/xfmname/mask-equality checks scattered through `Volume2D.raw`,
  `VolumeRGB.__init__`, and `Vertex2D.__init__`.
- **The only place that touches `db`.** Today `VertexData.__init__` calls
  `db.get_surf` for *every* `Vertex` you construct; a cache keyed on
  `(subject, xfmname)` makes that free.
- **Owns validation**: `space.validate(data)` replaces `VolumeData._check_size`
  and `VertexData._set_data` (including the left/right zero-padding).
- **Owns domain-specific rendering hooks**, so consumers stop branching:
  - `space.to_dense(data)` — today's `.volume` / `.vertices`
  - `space.pack_for_webgl(data)` — today's five branches in `webgl/data.py`
  - `space.sample_flatmap(data, ...)` — today's `hasattr(x, "xfmname")` branch
  - `space.write_hdf(h5)` / `Space.from_hdf(node)`

### 3.2 `BrainData` — one concrete class, not a hierarchy

```python
class BrainData:
    data: np.ndarray
    space: Space
```

`VolumeData` and `VertexData` collapse into this. The volume/vertex distinction
becomes `braindata.space`. This class owns: the array, the `movie` flag, hashing
and `.name`, the numpy operator overloads, and `copy(newdata)` — which becomes
trivially correct, since the space is carried over unchanged instead of being
reconstructed from a hand-written argument list per subclass.

### 3.3 `Dataview` — how data becomes color, composition only

```python
class Dataview:                  # cmap/vmin/vmax/description/state/attrs — ONE __init__
class DataviewScalar(Dataview):  # holds one BrainData          (today: Volume, Vertex)
class Dataview2D(Dataview):      # holds two BrainData          (today: Volume2D, Vertex2D)
class DataviewRGB(Dataview):     # holds r, g, b, alpha         (today: VolumeRGB, VertexRGB)
```

A `Dataview` **never inherits** `BrainData`. The duplication in §1.4 disappears
because there is one `__init__` for the view fields. The `_cls` hack in §1.5
disappears because the composed objects are all the same class. `.raw` always
returns a `DataviewRGB`, so consumers stop caring which subclass they were handed.

`view.space` is a passthrough to its data's space (with a check that all
components share one).

### 3.4 The dispatch rule

> **Consumers dispatch on the space, never on the view class.**

```python
# before, webgl/data.py — 5 sites, each needs a new branch per domain
if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
    encdata = brain.vertices
else:
    encdata = brain.volume

# after — 0 sites need editing when a domain is added
encdata = brain.space.pack_for_webgl(brain)
```

The N×M matrix becomes N + M.

### 3.5 The public API stays a set of thin constructors

`Volume`, `Vertex`, `Volume2D`, `Vertex2D`, `VolumeRGB`, `VertexRGB` keep their
names and signatures. They become thin: build a space, build the data, return the
right view. They carry no logic of their own.

### 3.6 Where iEEG lands

Everything below already exists after the refactor:

```python
elec = Electrodes(values, subject, coords, names=["STG1", ...], devices=[...])
elec2d = Electrodes2D(amplitude, coherence, subject, coords, cmap="PU_RdBu_covar")
cortex.webgl.show(elec)
```

The **only** genuinely new code is:

1. `ElectrodeSpace` (coords + metadata + validation + HDF serialization)
2. `ElectrodeSpace.pack_for_webgl`
3. a JS renderer for point/sphere geometry in `dataset.js` / `mriview.js`
4. `Electrodes*` constructor functions (a few lines each)

vmin/vmax, movies, 2D colormaps, RGB, `.raw`, `Dataset`, HDF save, `quickflat`
colorbars — all inherited for free, because the view layer knows nothing about
geometry.

---

## 4. Migration

### 4.1 Compatibility stance — decided: **A, in-place and compat-preserving**

Public classes keep their names *and* remain classes. `isinstance(x, cortex.Volume)`
keeps working, via compat shims that forward `.data`, `.xfmname`, `.mask`,
`.volume`, `.map()` to the composed parts. `cortex.Volume` is the most-used name
in the library, appears in every tutorial, and is `isinstance`-checked in user
code we cannot see — so nothing about the public surface moves.

The cost is a shim layer and a period of double bookkeeping. Every shim gets an
explicit deprecation marker and a target release, so the layer is temporary by
construction rather than by intention.

*(Considered and rejected: a new API alongside deprecation warnings, which keeps
two APIs live for a release cycle; and a clean break at 2.0, which breaks
downstream code for a large install base.)*

### 4.2 Sequencing — each step independently shippable, tests green throughout

**Step 0 — characterization tests, before touching anything.**
`test_braindata.py`, `test_dataset.py`, and `test_webgl_data.py` exist but do not
cover the full matrix. Add a parametrized round-trip test over
*every public class × {HDF save/load, `to_json`, `.raw`, `quickflat`}*. This is
the safety net that makes the rest of the work honest, and it will likely surface
one or two live bugs on its own.

**Step 1 — extract `Space`, no behavior change.** `VolumeData` / `VertexData`
keep their names and public surface but delegate shape/mask/hemisphere logic to a
space object. Pure refactor, no consumer changes.

**Step 2 — break the cooperative kwargs chain.** Give `Dataview.__init__`
explicit named parameters. Have `Volume.__init__` / `Vertex.__init__` *construct
and assign* rather than `super()`-chain through the data classes. Reject unknown
kwargs instead of sinking them into `attrs`. Fixes §1.1 and lets `_from_hdf_data`
drop its kwarg filtering.

**Step 3 — unify view construction.** `Dataview2D` and `DataviewRGB` call
`Dataview.__init__` instead of duplicating it. Normalize vmin/vmax defaulting to
one rule (§1.8) — recommend eager resolution at construction, so `.vmin` is
always a number.

**Step 4 — flip consumers to space-based dispatch.** `webgl/data.py`,
`quickflat/utils.py`, `quickflat/view.py`, `export/save_views.py`, `mapper/`,
`volume.py`, `blender/`. Behavior-preserving; this is where the §1.6 matrix
collapses.

**Step 5 — collapse `VolumeData` / `VertexData` into one `BrainData`.** Old names
become aliases. Public constructors become thin.

**Step 6 — add `ElectrodeSpace`**, its validation, HDF serialization, and
`pack_for_webgl`. The JS renderer is *not* part of this pass (§4.4), so step 6
lands the Python side and `quickflat` marker rendering; the WebGL viewer gains
electrode support in follow-on work.

### 4.3 Scope — decided: **Python only for this pass**

`cortex/dataset` and its Python consumers (`webgl/data.py`, `quickflat/`,
`export/`, `mapper/`, `volume.py`, `blender/`). `dataset.js` and `mriview.js`
keep their current `VolumeData` / `VertexData` split; the electrode renderer is
separate follow-on work.

One consequence to respect in **step 4**: `Space.pack_for_webgl` is now the
boundary between a refactored Python side and an unrefactored JS side. It must
keep emitting exactly what `dataset.js` expects today — the mosaic-PNG texture
path for volumes and the vertex-attribute path for surfaces, including the
premultiplied-alpha asymmetry documented in
[webgl/data.py:39-63](cortex/webgl/data.py:39). Treat that packer as a
compatibility surface with its own tests, not as a place to tidy up.

### 4.4 Deletions to fold in

Do these early — they shrink the surface being refactored:

- `Multiview` (dead, raises on construction)
- `Dataview.from_hdf`'s multi-view branch (dead)
- `blend_curvature` (already deprecated, wired in by hand in three places)

---

## 5. Decisions and open questions

**Decided:**

1. **Compatibility** — in-place, compat-preserving; public classes unchanged (§4.1).
2. **Scope** — Python only for this pass; JS viewer unchanged (§4.3).

**Still open — all three concern electrodes, and 4 and 5 should be settled before
step 6:**

3. **What space are electrode coordinates in?** sEEG coordinates typically come
   from a post-implant CT coregistered to the pre-op T1, so they are naturally in
   anatomical/magnet space — but users will also have them in MNI, or in a
   functional xfm's space. Proposal: `ElectrodeSpace` stores an explicit
   `coord_space` field rather than assuming, and conversion is a method.
4. **Should montages live in the filestore?** Today surfaces and transforms are
   database objects, and data objects reference them by name. Electrodes could
   follow that pattern (`db.get_electrodes(subject, montage)`) so several
   datasets share one montage and coordinates are not duplicated in every HDF
   file — or they could travel with the data object. This changes serialization,
   so decide it **before** step 6.
5. **Do electrodes need surface/volume interop?** e.g. "nearest vertex to each
   electrode", or smearing an electrode's value onto nearby cortex. If yes, that
   is a `Mapper` between spaces — which the space abstraction supports naturally,
   but it should be an explicit goal rather than an accident.
