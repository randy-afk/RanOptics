# Changelog
All notable changes to RanOptics will be documented here.
Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

## [2.1.1] - 2026-08
### Fixed
- Tao/Bmad standalone build: fixed a library version conflict where a staged dependency (e.g. `libcurl.so.4`) could resolve against PyInstaller's own bundled copy of a same-named library (e.g. an older `libssl.so.3`, bundled for Python's `ssl` module) instead of the correct staged one, causing a symbol-version error at load time. Every staged Bmad library is now explicitly preloaded before `libtao.so` itself, so the correct versions are already resident by the time anything looks for them.

## [2.1.0] - 2026-08
### Added
- Tao/Bmad backend now works in the standalone packaged executable. pytao is bundled directly; the actual compiled Bmad library (`libtao.so`/`.dylib`/`.dll`) is pointed at explicitly via two new fields in the GUI (shown when Tao is selected):
  - **Bmad library**: path to your installed `libtao.so`/`.dylib`/`.dll`.
  - **Extra library dirs**: optional, comma-separated directories for libtao's own dependencies (GSL, LAPACK, FFTW3, HDF5, etc.), for installs where those aren't sitting next to the library itself (e.g. a hand-built Bmad, or one assembled from system packages instead of a single conda environment).
  
  Both are only needed inside the packaged executable — running from source with pytao/Bmad already on your environment needs neither.
- Backend-availability check extended to validate the Bmad library path (and any extra dirs) exist before running.
### Fixed
- The Bmad library path mechanism above replaces an earlier, non-working approach: setting `LD_LIBRARY_PATH` from inside the running app did nothing (glibc reads and caches that variable once at process start, before any Python code runs — confirmed by testing the identical scenario with the change made mid-process vs. genuinely before the process starts). Library loading is now done by staging symlinks into one directory and loading from there, which is the mechanism confirmed to actually work via `dlopen()`'s own same-directory dependency search — verified end to end against a real Bmad lattice in an actual frozen build, not just an unfrozen sanity check.
### Changed
- ubuntu-latest CI runner is now Ubuntu 24.04; `libgl1-mesa-glx` (removed in 24.04) replaced with `libgl1` in the build workflow.
- Build workflow matrix jobs no longer cancel each other on one platform's failure (`fail-fast: false`), and now have explicit `contents: write` permission for uploading release assets.
- kaleido pinned to 1.2.0 in the build workflow: 1.3.0 has a submodule that crashes PyInstaller's dependency collection.

## [2.0.0] - 2026-08
### Fixed
- Expression evaluator (`expr.py`) sandbox escape — dunder attribute access (e.g. `().__class__.__bases__`) could reach arbitrary Python classes from a preset file. Eval namespace is now AST-validated before evaluation.
- Expression evaluator: list/dict comprehensions raised `NameError` due to a split globals/locals eval scope; also fixed a `/`-containing namespace key substitution that used blind substring replace instead of word-boundary matching.
- Floor plan builder (`panels.py`): `KeyError` when floor coordinates exist for some but not all elements (e.g. MAD-X survey file shorter than the twiss table, or Tao universe/patch gaps).
- Floor plan builder: `flip_bend` did not negate `flr_theta1`, producing incorrect dipole curvature when survey/`.flr` data included a per-element exit angle.
- Floor plan builder: YZ dead-reckoning legend ignored `show_fp_legend` and could duplicate the "Dipole" legend entry.
- Twiss panel auto-tick logic (`engine.py`) crashed with `NaN → int` when all plotted beta values were zero.
- Grid layout (`engine.py`): multi-universe floor-plan cells dropped universe 0's legend keys, leaving its legend unpositioned/unhideable.
- Grid layout: `latdiff` panel rows all showed a duplicate title instead of only the first of each 3-row block.
- xsuite loader: bend-angle sign was inverted when reading `.angle` after `.h`, mirroring every dipole's curvature in xsuite lattices.
- MAD-X loader: `VKICKER` elements displayed `hkick` instead of `vkick` in hover text (all kicker subtypes collapsed to one internal key).
- GUI: CSV export left the status bar stuck on "Exporting CSV…" on failure and had no re-entrancy guard against overlapping exports.
- GUI: loading a preset only overlaid keys present in the file, so fields missing from an older/partial preset silently kept whatever was on screen. Preset load now resets to app defaults first.
- GUI: expression panel composer connected a closure to the global `QApplication.focusChanged` signal on every open and never disconnected it, leaking references to destroyed widgets on repeated open/cancel.
- GUI: Dry Run's lightweight loader dispatch had no MAD-X branch and fell through to `load_elegant()` on `.tfs` input; `load_madx` wasn't even imported in `gui.py`.
### Changed
- Hardcoded colors in `overlays.py` and `gui.py` replaced with `themes.py` tokens throughout, per project convention.
- Removed dead code: unused `_WorkerThread`/`_FodoLogo` classes and the unreferenced `_RanOpticsLogo` import in `gui.py`.
### Added
- MkDocs Material documentation site (`docs/`, `mkdocs.yml`) with GitHub Pages deploy workflow.
- `LICENSE` (MIT), `requirements.txt`, `environment.yml`.
- Standalone executable builds via PyInstaller + GitHub Actions (`.github/workflows/build-ranoptics.yml`), triggered on GitHub Release creation. Bundles PySide6/plotly/numpy/kaleido; physics backends (Tao/Bmad, ELEGANT, MAD-X) remain external/user-installed, same as the source install.
- Backend-availability check (`check_backend_ready()`) before Run/Dry Run/Export CSV — shows a clear popup instead of a raw traceback when the selected backend (Tao, ELEGANT, xsuite) isn't installed or on `PATH`.

## [1.3.0] - 2026-05
### Added
- Colors tab in right panel — per-element-type color customization
- Color picker with swatch buttons and reset-to-default for each element type
- Element colors persisted in presets (saved to JSON)
- elem_colors parameter added to plot_optics() and threaded through all panel builders
---
## [1.2.2] - 2026-05
### Fixed
- Dipole polygon bend angle wrapped to [-π, π] to fix circle shapes in multi-universe plots
- Horizontal bends now draw as flat rectangles in YZ floor plan
- srange filter now correctly applies to floor plan panels (XZ and YZ)
- Floor plan YZ x-axis label corrected from "s (m)" to "Z (m)"
- ELEGANT loader: added flr_theta1 for correct dipole polygon bend shape
- xsuite loader: added flr_theta1 for correct dipole polygon bend shape
- Floor plan s-axis label no longer stamped on floor plan rows when bar panel is absent
---
## [1.2.1] — 2026-05
### Fixed
- **Y-axis alignment** across all panels — secondary y-axis (Dispersion on Twiss panel) had `domain=None` causing each subplot to independently size its left boundary. Fixed by copying domain from primary to secondary y-axes after layout is applied.
- **Expression panel zoom** — expression panels were not linked to the shared x-axis reference (`matches=ref`), so zooming any other panel did not zoom expression panels. Fixed.
- **Custom/expr panel height independence** — two custom panels or two expression panels shared the same height entry (keyed by type string). Each panel now gets a unique `_id` (UUID) injected at creation time, enabling fully independent height control per panel slot.
- **`_panel_px` spec key lookup** — custom panel dicts were being looked up under an empty string key instead of their type. Fixed to correctly resolve `'custom'` or `'expr'` from the panel dict.
- **Missing imports** — pre-existing `NameError` bugs: `_read_tfs` missing from `engine.py` and `overlays.py`, `load_xsuite` missing from `overlays.py`, `load_tao`/`load_elegant`/`load_xsuite`/`_parse_tao_init` missing from `gui.py`. All fixed.
- **File dialog theme** — native file dialogs used the system light theme (black text on white). Switched to Qt-rendered dialogs (`DontUseNativeDialog`) with full RanOptics dark stylesheet applied globally via `QApplication.setStyleSheet`.
- **Duplicate code removed** — 730 lines of duplicate function definitions removed across `loaders.py`, `panels.py`, and `engine.py`. `expr.py` is now the canonical home for expression evaluator functions.

### Added
- **Log panel: timestamps** — every log line is prefixed with `[HH:MM:SS]`.
- **Log panel: deduplication** — consecutive identical lines collapse to `↑ repeated N×` instead of flooding the log (e.g. Tao sanity check errors that repeat 100+ times).
- **Log panel: filter dropdown** — filter log to All / Warnings+ / Errors only.
- **Log panel: auto-scroll toggle** — pin button to pause auto-scrolling while reading mid-log.
- **Log panel: copy button** — copies full log contents to clipboard.

### Changed
- `TEAL` color alias renamed to `HIGHLIGHT` (was misleadingly named — the color is warm yellow `#FEC868`, not teal).
- `FONT_SMALL` corrected from 11pt to 9pt (was identical to `FONT_MAIN`).
- `SUCCESS` and `PEACH` are now proper Python aliases for `RAN_CLR` and `ACCENT` respectively, preventing silent color divergence.
- Logo sine curves updated from orange (`ACCENT`) to green (`RAN_CLR`) to match the "Ran" text color.
- Author/support lines updated: `Author: Randika Gamage (randika@jlab.org)` / `Support: ¯\_(ツ)_/¯  (good luck, I believe in you)`.
- Version bump: `v1.2.0` → `v1.2.1`.

---
## [1.1.0] — 2026-05
### Added
- **Beamline Bar Lite** toggle in Panel Options (off by default). When enabled, the beamline bar uses the same two-trace rendering method as the floor plan — one invisible hover line plus one filled polygon per element, laid out linearly along s. This significantly reduces plot generation time for large lattices. Zoom sync with other panels is preserved. Element info is shown in the optics panel hover tooltips instead of the bar itself.
- **RF cavity ovals** in beamline bar — both standard and lite mode now render RF/L-cavities as ovals scaled to the element length, matching the floor plan appearance.
### Changed
- Element name now appears in optics panel hover tooltips when Beamline Bar Lite is active.
---
## [1.0.0] — 2026-04
### Initial release
- PySide6 GUI with tabbed layout (Input, Beam Settings, Panels, Visual, Export)
- Backends: Tao (Bmad), ELEGANT, xsuite, MAD-X (TFS files)
- Panel system: Twiss, Beta, Alpha, Dispersion, Orbit, Phase Advance, Beam Size
- Floor plan panels: X-Z and Y-Z with tunnel wall overlay
- Beamline bar panel with element labels and wildcard annotations
- Lattice Summary table (tune, chromaticity, element counts per universe)
- Lattice Diff panel — element-by-element comparison (strengths, entry/exit positions)
- Custom panels — mix any standard quantities on Y1/Y2 axes
- Expression panels — user-defined Python expressions plotted as optics panels
- Multi-universe support (Tao) — overlay or compare universes
- Compare mode — overlay, separate, difference, difference (%)
- Beam size calculation — geometric or normalized emittance, n·σ envelope
- Per-panel pixel height controls
- Configurable panel spacing
- CSV export — one file per panel
- Preset save/load system
- Recent files menu
- Session-friendly: config persists between runs via ~/.ranoptics_recent.json and ~/.ranoptics_presets.json
- Interactive HTML output via Plotly
- Optional PNG/PDF export via kaleido
