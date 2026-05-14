# Interface Overview

![RanOptics GUI](assets/screenshot.png)

The RanOptics window is divided into three main areas.

---

## Left panel — Input & Settings

### Input tab

| Field | Description |
|-------|-------------|
| **Input file** | Path to your simulation file. Auto-detects backend from extension. |
| **Code backend** | Override the auto-detected backend if needed. |
| **Output HTML** | Path for the output plot file. |
| **Layout mode** | `panels` (stacked panels) or `floor` (floor plan only). |
| **Range START:END** | Limit the plot to a sub-range by element name or s position. |
| **Tunnel wall file** | Optional `.dat` file with tunnel wall coordinates for floor plan overlay. |
| **Compare files** | Add additional files to compare against the primary. |
| **Compare mode** | Overlay, Separate, Difference, or Difference (%). |
| **Normalize s** | Map s → [0, 1] to align lattices of different lengths. |

### Beam Settings tab

Configure emittance, beam energy, and other beam parameters used for beam size calculations.

---

## Right panel — Panels / Visual / Export

### Panels tab

Add, remove, and reorder plot panels. Each panel has:

- A **height field** (pixels) — controls how tall that panel is in the output
- **▲▼** buttons — reorder panels
- **×** button — remove the panel
- **⋮** button — annotation pattern (wildcard element labels, e.g. `IPM*`)
- **÷** button — legend position override

The **Panel Options** section contains global settings:

| Option | Description |
|--------|-------------|
| **Show tune/chromaticity** | Annotates Qₓ, Qᵧ, Qₓ', Qᵧ' on the first data panel |
| **Show panel titles** | Toggle panel title text |
| **Beamline bar lite** | Faster rendering for large lattices |
| **Panel spacing** | Vertical gap between panels in pixels |

### Visual tab

Controls for floor plan display, element rendering, color themes, font sizes, and legend positions.

### Export tab

PNG and PDF export settings including DPI and figure dimensions.

---

## Bottom bar

| Button | Description |
|--------|-------------|
| **▶ Run** | Run the simulation and generate the plot |
| **■ Cancel** | Cancel a running job |
| **🌐 Open Plot** | Open the output HTML in your browser (lights green when ready) |
| **Dry Run** | Validate input without generating a plot |
| **Export CSV** | Export panel data as CSV files |
| **Clear log** | Clear the output log |

The **Output Log** shows timestamped progress, warnings, and errors. Use the filter dropdown to show All / Warnings+ / Errors only. The **Auto** button toggles auto-scrolling, **Copy** copies the full log to clipboard.
