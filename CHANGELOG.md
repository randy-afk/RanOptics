# Changelog
All notable changes to RanOptics will be documented here.
Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

## [2.2.1] - 2026-08
### Fixed
- **macOS standalone build: Tao/Bmad now loads.** Reported from a real macOS test of 2.2.0: `Symbol not found: _EVP_MD_CTX_get_size_ex, referenced from libs2n.1.0.0.dylib, expected in _MEIxxxx/libcrypto.3.dylib`. Same root cause as the Linux fix in 2.1.2 — Bmad's dependency chain resolves against PyInstaller's own bundled OpenSSL, which is older than the conda-forge OpenSSL Bmad was built against. The OpenSSL bundling step is now applied to macOS as well as Linux, using `libssl.3.dylib`/`libcrypto.3.dylib` and selecting the micromamba build from the runner's architecture (`osx-arm64` or `osx-64`).
- Build now asserts that the OpenSSL it is about to bundle actually carries the symbol each platform needs (`OPENSSL_3.2.0` on Linux, `EVP_MD_CTX_get_size_ex` on macOS) rather than trusting the version pin. A version range alone proved insufficient: conda-forge's newer OpenSSL requires `__osx >=11.0`, and a solve that cannot see the real macOS version silently falls back several minor versions to a build missing the needed symbol. The build now fails loudly at that point instead of producing a binary that only breaks when a user loads Bmad.

## [2.2.0] - 2026-08
### Added
- **App themes.** Five colour themes (Petrol, Classic Green, Sulfur Sea, Ultraviolet, Oxblood), each with matched light and dark palettes. Picker sits in the header beside the existing light/dark toggle; both apply live, no restart. Theme and mode persist between sessions. Petrol is the new default. Every palette assigns colours to fixed roles (primary action, secondary action, section label, alert) rather than using them decoratively, which is what keeps a multi-colour theme readable.
- **Bmad library paths are remembered.** The "Bmad library" and "Extra library dirs" fields save to `~/.ranoptics_settings.json` and are restored automatically at startup, so they no longer have to be re-entered every launch.
- **Curve smoothing.** Optional spline interpolation of optics curves ("Smooth curves" in Appearance → Display, with an adjustable amount). Display-only and off by default: it interpolates between computed points without any knowledge of the physics, so it can overshoot near a waist and rounds off genuine discontinuities. Floor-plan and beamline-bar geometry is never smoothed, and CSV export is unaffected.
### Changed
- **Bmad staging now follows the real dependency closure.** Previously every shared library in the source directories was staged, which on a typical conda install meant ~1200 files for a library that needs 18, including 240 Qt6 libraries pulled into a process already running PySide6. `libtao.so`'s `DT_NEEDED` entries are now walked recursively and only what's actually required is staged, about 39 files. Parsed directly from the ELF header rather than shelling out to `ldd`/`objdump`, which aren't guaranteed present on a machine running the packaged binary. Non-ELF platforms (macOS `.dylib`, Windows `.dll`) keep the previous behaviour.
- **Bmad paths are no longer stored in presets.** They're machine-specific, so a preset containing a local library path was meaningless on another machine, and loading any preset would wipe the saved path. They now live only in the settings file.
- Documentation site restyled to a Petrol-family palette, deliberately offset from the application's own so screenshots keep a visible edge against the page. Content images now get an explicit border and shadow.
### Fixed
- **Staging directories no longer accumulate.** `_stage_bmad_lib` created a fresh temp directory on every Tao load and never removed it, leaving one behind per Run click indefinitely. Staging now reuses one stable directory per library/extra-dirs combination, rebuilt automatically if the install moves or a symlink breaks.
- Light-mode contrast: white button text on the green accent scored 3.77 against the 4.5 minimum for body text, and section labels failed on three palettes. All themes now pass 4.5 in both modes for button text, section labels, help text and body text.
- The light GUI palette was previously only reachable through the header toggle and had no counterpart for any other theme; every theme now has a proper light sibling.

## [2.1.3] - 2026-08
### Fixed
- Linux standalone build: the OpenSSL bundling step added in 2.1.2 failed in CI, so no Linux executable was produced. conda-forge's `openssl` package now resolves to 4.0.x, which ships `libssl.so.4`/`libcrypto.so.4`, but the build specifically needs SONAME `libssl.so.3` — that's what both Python's `_ssl` and Bmad's `libcurl.so.4` are linked against, and a 4.x copy satisfies neither. Pinned to `openssl>=3.2,<4` (currently 3.6.3, which carries the `OPENSSL_3.2.0` symbols libcurl needs) and added explicit existence checks so any future mismatch fails at the fetch step with a clear message instead of part-way through the build. No application code changed from 2.1.2.

## [2.1.2] - 2026-08
### Fixed
- Tao/Bmad standalone build: the v2.1.1 preload fix didn't actually work — root-caused by directly inspecting a real failing process. Python's own `ssl`/`hashlib` modules load PyInstaller's bundled `libssl.so.3` (capped at symbol version `OPENSSL_3.0.x`, from the CI runner's system OpenSSL) automatically, before the app window even appears, which permanently claims that library's SONAME for the whole process. By the time Run is clicked, no amount of preloading a newer copy for Tao's benefit can dislodge it, since the dynamic linker already resolved that SONAME. Confirmed by direct reproduction (forcing the exact load order) and fixed at the source: the Linux build now bundles conda-forge's own OpenSSL (the same distribution Bmad itself is built against) in place of the runner's system copy, so whichever loads first is already new enough for both Python and Bmad. Verified end to end against a real Bmad lattice in an actual frozen build, including deliberately forcing `ssl` to load first to reproduce the exact failure ordering.

## [2.1.1] - 2026-08
### Fixed
- Tao/Bmad standalone build: fixed a library version conflict where a staged dependency (e.g. `libcurl.so.4`) could resolve against PyInstaller's own bundled copy of a same-named library (e.g. an older `libssl.so.3`, bundled for Python's `ssl` module) instead of the correct staged one, causing a symbol-version error at load time. Every staged Bmad library is now explicitly preloaded before `libtao.so` itself, so the correct versions are already resident by the time anything looks for them. **Note: this fix did not actually resolve the issue on real-world installs — see 2.1.2.**

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
