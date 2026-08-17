# Working branches and the order to land them

Notes for moving the in-progress type-annotation work onto `main`, or onto each
other. Read this before rebasing: what each branch carries, what was resolved
against upstream, and which choices to keep the next time it is replayed.

## The stack

```
main @ 00c5ec83a (#688, dataset docstrings)
 └── types-base ................. the part of types-orig upstream has not merged  9 commits
      └── types-easy ............ module-by-module mypy fixes, and the runtime bugs they found  24
           └── types-data ....... dataset restructure, unions + TypeIs narrowing  9
                └── types-data-protocols  dataset restructure, spatial ABCs  <-- current  59
```

This is one linear stack: each branch contains the whole of the one above it.

It sits on `00c5ec83a`, not the tip of `main`. These upstream commits came after
it and are not in the stack yet:

- `1af836bff` FIX unify NaN and alpha handling across quickflat, WebGL and RGB dataviews (#695)
- `a84476df0` Build(deps): bump astral-sh/setup-uv from 10.1.0 to 10.2.0 (#755)
- `31e3486da` NF bumpy flatmap redesign (#720)
- `9d2e47143` NF webgl views animation gui (#752)
- `1e4c0790b` NF optimize 4D timeseries performance and add interactive timeseries panel (#745)

#695 and #745 change `cortex.dataset`, so they will need porting into the
restructure (see below); #720 and #752 overlap `brainctm.py` and `webgl/view.py`.

`types-base` holds what was left of `types-orig` once upstream had squash-merged
the rest of it as #666, #667, #669 and #682: annotations for `mapper`,
`polyutils`, `freesurfer`/`segment`, `svgoverlay`, `brainctm`/`webgl` and `utils`,
an `openctm.pyi` stub, and one behavior fix (`facepick.js`: clicking a vertex
reaches the Python `pickerfun` again). `types-orig` itself is kept only as a record
of the original history; nothing is built on it.

`types-easy` fixes mypy errors one module at a time, outside `cortex.dataset`, and
fixes the runtime bugs that typing turned up (`flatten_slim` always returning
`None`, the `sha1(str)` crash, `export/panels`' `zip(*...)` unpacking,
`get_texture` returning `None` when given `name`, and others). It also narrows
`Transform.reference` and adds `Transform.reference_nifti`. Two public changes:
`get_mapper(type=...)` is now `get_mapper(maptype=...)`, so it stops shadowing the
builtin, and `Database.get_surf` takes `merge`/`nudge` keyword-only.

`types-data` is the first nine commits of the restructure. It ends with the
volumetric/surface distinction expressed as closed unions narrowed by `TypeIs`
predicates. `types-data-protocols` continues from there and replaces that narrowing
with abstract bases per spatial axis. See `cortex/dataset/TYPING_ALTERNATIVES.md`
for what was tried and why. `types-data` is a stopping point, not a competing
design: landing `types-data-protocols` lands its commits too.

## Order of operations

1. **Rebase `types-base` onto `main` first.** Everything above it sits on top of
   it, so rebasing a branch before its base only creates conflicts twice.
2. **Then rebase the rest of the stack in one pass.** From `types-data-protocols`,
   `git rebase --update-refs --onto <rebased types-base> <old types-base>` replays
   `types-easy`, `types-data` and `types-data-protocols` together, and moves all
   three branch refs. Add a `break` after `types-data`'s last commit in an
   interactive rebase if you want to run mypy and the tests at that point.
3. **Port upstream's `cortex.dataset` changes into every restructure commit.**
   Upstream edits code that the restructure moves between files, so these
   conflicts do not resolve hunk by hunk. What worked: at each stop, take the
   commit's own version of the conflicting `cortex/dataset` files, then reapply
   upstream's changes wherever that commit put the code. Each commit then carries
   them, and none regresses upstream between the first restructure commit and the
   last. `git rebase --exec` with a check that the port is still complete catches
   a commit that silently loses part of it.

## Upstream changes ported into the restructure

From `types-orig`'s base to `00c5ec83a`, upstream changed `cortex.dataset` in #666,
#669, #681, #688, #701 and #716. Each commit of `types-data` and
`types-data-protocols` carries them in the restructure's layout:

- **#716** low-frequency `random()`: the default for `Volume` and `Vertex`, with
  `random_type='uniform'` for plain gaussian noise. Once `empty`/`random` take
  their shape from the space, the uniform branch goes through `_sample` and
  `template_shape`, and `_low_freq_noise` accepts any shape tuple.
- **#688** docstrings, on whichever method now holds the code. Where the
  restructure later writes its own docstring or comment for the same thing
  (`Dataset.uniques`, the `get_surf` fallback overload, the single
  `ScalarView.to_json` and `DataviewRGB.to_json` that replace the per-class
  overrides), the restructure's text wins. Once the constructors drop
  `**kwargs`, #688's list of webgl keys moves under `attrs=`, and `state` gets
  its own entry as a parameter.
- **#701** Windows-safe temp file in `Dataset.get_overlay`.
- **#666/#669** `h5py.File` rather than `h5py._hl.files.File`, and unquoted
  `Dataset` annotations, in `dataset.py`.
- **#681** is not ported: `ScalarView._resolve_percentiles` already keeps the
  `np.float64` that #681's `.astype(float)` produces.

Outside `cortex.dataset`, keep these resolutions:

- Quickflat signatures take upstream's `fig: Axes` and wider return types with
  the restructure's parameter types (`HasSubject`, `RenderableView`).
- `make_gif` returns `Union[str, BinaryIO]`: upstream returns the destination,
  and the restructure had annotated `None`.
- The float32 fix for saving views (`_dumps`) applies where upstream now
  serializes the viewer metadata, in `html.generate(data=...)`.
- Upstream's `TestAddData.test_rejects_unknown_subject` assigned
  `view.subject`, which the restructure makes read-only. It sets `_subject` from
  the collapse commit on, and `space.subject` once views have a space.

## Where `types-easy` and the restructure overlap

`types-easy` was written alongside the restructure, not on top of it, and the two
were stacked afterwards. Keep these choices if those commits are ever replayed
separately again:

- **`braindata.py`**: the restructure moved its contents into `views.py`, so
  `types-easy`'s `xfm.reference_nifti.affine` change lives in the moved code
  (`Volume.save_nii`).
- **`get_surf`**: both made `merge` and `nudge` keyword-only in every signature.
  The restructure's longer comments were kept.
- **`Database.get_cache`**: `self.auxfile._require_h5().filename` was kept over
  `h5.filename` with a `# type: ignore`.
- **`blender.add_cutdata`**: the restructure's `Vertex`/`Volume` dispatch was kept,
  with the assert that `vmin`/`vmax` are set (views resolve them at init) rather
  than `types-easy`'s `ValueError`.
- **`export/save_views.py`**: both kept: `types-easy`'s
  `interpolation_params: ViewParams` declaration and the restructure's
  `isinstance(volume, VolumetricView)`.
- **`webgl/data.py`**: `types-easy`'s `_pack_png` was dropped. The restructure moved
  it to `dataset/_webgl.py` as `pack_png`, which already has the same `tobytes()`
  fix.

## Gotchas

- **The visual regression tests are upstream's now** (#672,
  `cortex/tests/test_visual_regression.py`). The `visual-regression-tests` branch
  and the cherry-picks of it that used to sit on `types-data-protocols` are
  obsolete. The reference images are a `main` baseline on purpose, so
  regenerating them is how a real regression gets blessed: read
  `cortex/tests/reference_images/README.md` before reaching for
  `REGENERATE_REFERENCE_IMAGES=1`.
- **Judge a rebase against its base, not against zero failures.** Some visual
  comparisons fail depending on the rendering environment, for example 14 at
  `00c5ec83a` in the dev container. A rebased branch should fail exactly the set
  its base fails.
