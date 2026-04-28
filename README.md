# RanOptics — Accelerator Optics Plotter

**RanOptics** is a self-contained desktop GUI for visualizing and analyzing accelerator optics and beam dynamics data. It produces interactive HTML plots via Plotly and supports multiple simulation backends.

> *"Run your optics."*

---

## Features

- **Multiple backends** — Tao (Bmad), ELEGANT, xsuite, MAD-X (TFS files)
- **Interactive HTML output** — pan, zoom, hover, exportable plots
- **Panel system** — stack any combination of panels in any order:
  - Twiss & Dispersion, Beta, Alpha, Dispersion, Orbit, Phase Advance, Beam Size
  - Floor Plan X-Z and Y-Z
  - Beamline bar with element labels
  - Lattice Summary table
  - Lattice Diff (compare two lattices element by element)
  - Custom panels (mix any quantities on Y1/Y2 axes)
  - Expression panels (user-defined Python expressions)
- **Multi-universe support** — overlay or compare multiple Tao universes
- **Compare mode** — overlay, separate, difference, or difference (%) between files
- **Floor plan** — X-Z and Y-Z views with tunnel wall overlay support
- **Beam size calculation** — geometric or normalized emittance, n·σ envelope
- **CSV export** — per-panel tabular data export
- **Presets** — save and reload GUI configurations
- **Recent files** — quick access to previously loaded files
- **Dark teal theme** — easy on the eyes for long sessions

---

## Requirements

### Python packages

```bash
pip install numpy plotly PySide6 kaleido
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate ranoptics
```

### Backend dependencies

| Backend | Requirement |
|---------|-------------|
| **Tao / Bmad** | [`pytao`](https://github.com/bmad-sim/pytao) — see pytao documentation for installation |
| **ELEGANT** | `elegant` and `sddsconvert` must be on your `PATH` |
| **xsuite** | [`xsuite`](https://xsuite.readthedocs.io) Python package |
| **MAD-X** | No Python package needed — run MAD-X yourself and point RanOptics at the `twiss.tfs` / `survey.tfs` output files |

---

## Usage

```bash
python ranoptics.py
```

1. Select your input file (`.init` → Tao, `.ele` → ELEGANT, `.json` → xsuite, `.tfs` → MAD-X)
2. The code backend is auto-detected from the file extension
3. Configure panels, beam parameters, and visual options
4. Click **▶ Run**
5. Click **🌐 Open Plot** when it lights up green

---

## Panels

| Panel | Description |
|-------|-------------|
| `Twiss & Dispersion` | βₓ, βᵧ (left axis), ηₓ, ηᵧ (right axis) |
| `Beta Functions` | βₓ, βᵧ |
| `Dispersion` | ηₓ, ηᵧ |
| `Alpha Functions` | αₓ, αᵧ |
| `Orbit` | Closed orbit x, y |
| `Phase Advance` | μₓ, μᵧ |
| `Beam Size` | σₓ, σᵧ from emittance and dispersion |
| `Floor Plan X-Z` | Horizontal floor plan |
| `Floor Plan Y-Z` | Vertical floor plan |
| `Beamline Bar` | Element layout bar |
| `Lattice Summary` | Tune, chromaticity, element counts |
| `Lattice Diff` | Element-by-element comparison (strengths + positions) |
| `Custom` | Any combination of standard quantities on Y1/Y2 |
| `Expression` | User-defined Python expressions (e.g. `k1 * beta_a`) |

---

## File Structure

```
ranoptics.py        # Single self-contained script — GUI + all backends
README.md
requirements.txt
environment.yml
LICENSE
CHANGELOG.md
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Randika Gamage**  
Jefferson Lab (JLab), Newport News, VA  
randika@jlab.org

*Support: Schrödinger's helpdesk — may or may not respond.*
