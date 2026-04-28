# Changelog

All notable changes to RanOptics will be documented here.

Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

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
