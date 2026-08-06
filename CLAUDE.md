# RanOptics: Project Guide for Claude

RanOptics is a desktop accelerator-optics plotting tool: a PySide6 (Qt) GUI that
builds Plotly figures. Always refer to the tool as RanOptics. Do not use any
earlier or alternate name for it.

## Stack
- Python, PySide6 (Qt) for the GUI
- Plotly for all plotting, rendered to `optics.html`
- Four simulation backends: Bmad/Tao, ELEGANT, MAD-X, xsuite

## Layout
- `RanOptics.py`: entry point
- `core/engine.py`: main `plot_optics()`, assembles the Plotly figure (largest module)
- `core/gui.py`: PySide6 GUI, widget wiring, `_collect_kwargs`, preset save/load
- `core/panels.py`: panel and floor-plan builders (`_build_floor_plan`,
  `_build_floor_plan_yz`, `_build_layout_bar`, tunnel walls)
- `core/loaders.py`: the four backend loaders (`load_tao`, `load_elegant`,
  `load_madx`, `load_xsuite`)
- `core/expr.py`: expression namespace and evaluation
- `core/overlays.py`: overlay dialogs
- `core/utils.py`: shared helpers (element-sizing primitives, hover text)
- `core/themes.py`: color palette and fonts (see conventions)
- `core/logo.py`: QPainter logo widget

## Backend gotchas (read before touching plotting)
- Some features are backend-specific and only appear when that backend is loaded.
  Multiple universes is Bmad/Tao only. Do not assume a field exists across all
  four backends.
- Survey/floor coordinates (`flr_x0`, `flr_y0`, `flr_z0`, and their `1` variants)
  are populated by ELEGANT (`.flr`), xsuite, MAD-X only when a survey file is
  supplied, and Tao only when floor data is extracted. MAD-X without a survey file
  and Tao without floor data fall back to dead-reckoning: the beampipe geometry is
  computed from element length and angle inside the floor-plan builder.
- Keep floor-plan sizing backend-independent. Element height and axis behavior
  derive from `s_start`/`length` plus whatever `flr_*` values exist, and must
  handle their absence. When there are no survey coords, leave the transverse (X)
  axis auto-ranged so the computed beampipe defines the extent. Do not assert an
  explicit X range from `flr_x` data that may not be present.

## Coding conventions
- GUI code: never hardcode colors or font sizes. Define them in `core/themes.py`
  and import from there. Do this automatically, without being asked.
- Keep changes minimal and localized. When refactoring, preserve existing variable
  names so downstream code keeps working.
- After edits, run `python -m py_compile core/*.py`. A headless check cannot run the
  GUI or the backends, so also state which visual smoke test is still needed
  (which backend, ring vs line, etc.).

## How to work with me (Randy)
- Diagnose first, propose the fix, then WAIT for my explicit go-ahead before editing
  any code. No exceptions.
- No unsolicited edits. If you notice something outside the requested change, flag
  it, do not edit it, unless I say otherwise.
- Do not make assumptions. If a request is unclear, ask one clarifying question
  before proceeding.
- Read the whole message, including pasted code, errors, and uploaded files, before
  responding.
- Be concise. No walls of text, no padded closers, no over-hedging.
- Honest pushback only. Correct factual errors directly, separate facts from
  judgment calls, and drop an objection once I give my reasoning. Do not manufacture
  disagreement.
- A one-liner from me often has nuance underneath. Ask before assuming the simple
  reading is my whole position.
- Answer the question I asked. Do not volunteer facts or caveats I already know.
- Writing style: do not use em dashes. Use commas, periods, parentheses, or
  restructure the sentence.
