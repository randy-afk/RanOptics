# Changelog

All notable changes to RanOptics will be documented here.

Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

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
