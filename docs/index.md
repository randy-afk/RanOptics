# RanOptics — Accelerator Optics Plotter

**RanOptics** is a self-contained desktop GUI for visualizing and analyzing accelerator optics and beam dynamics data. It produces interactive HTML plots via Plotly and supports multiple simulation backends.

> *"Run your optics."*

![RanOptics GUI](assets/screenshot.png)

---

## What it does

RanOptics takes your lattice simulation output and turns it into interactive, publication-quality optics plots — directly in your browser, with no additional software required.

- Stack any combination of panels in any order
- Pan, zoom, and hover over any element or data point
- Compare multiple lattices side by side
- Export to HTML, PNG, or CSV

---

## Backends

| Backend | Input file | Notes |
|---------|-----------|-------|
| **Tao / Bmad** | `.init` | Requires `pytao` |
| **ELEGANT** | `.ele` | Requires `elegant` and `sddsconvert` on PATH |
| **xsuite** | `.json` | Requires `xsuite` Python package |
| **MAD-X** | `.tfs` | No Python package needed — point at `twiss.tfs` output |

The backend is auto-detected from the file extension.

---

## Quick links

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Panel types](panels.md)
- [Expression panels](expressions.md)
- [Changelog](changelog.md)
