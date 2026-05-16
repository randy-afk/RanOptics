# Changelog
All notable changes to RanOptics will be documented here.
Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`
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
