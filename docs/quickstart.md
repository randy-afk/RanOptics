# Quickstart

This guide walks through a complete RanOptics session from launching the GUI to viewing your interactive plot.

---

## 1. Launch the GUI

From your terminal, navigate to the RanOptics directory and run:

```bash
python ranoptics_gui.py
```

The Output Log at the bottom will display a `Ready` message once the application has initialized successfully.

---

## 2. Load a Lattice File

In the **Input** tab, click **Browse** next to the *Input file* field and select your lattice file. RanOptics auto-detects the backend from the file extension:

| Extension | Backend |
|---|---|
| `.init` | Tao / Bmad |
| `.ele` | ELEGANT |
| `twiss.tfs` + `survey.tfs` | MAD-X |
| `.json` | Xsuite |

The *Code backend* field will update automatically. You can override it manually if needed.

Specify an output filename in the *Output HTML* field (e.g. `optics.html`). This is the interactive plot file that will be generated.

---

## 3. Configure Plot Settings

Still in the **Input** tab, under **Plot Settings**:

- **Layout mode** — choose `panels` for a standard stacked optics dashboard, or `floor` for a floor plan only view.
- **Range** — optionally restrict the plot to a sub-section of the lattice. You can specify element names (e.g. `QUA01:QUA06`) or s-positions (e.g. `3.0:19.0`).
- **Tunnel wall** — if you have a tunnel geometry file (`.dat`), browse for it here to overlay the physical enclosure on the floor plan.

---

## 4. Add Panels

Switch to the **Panels** tab. Click any preset button to add it to the active plot stack:

- **Twiss Dispersion**, **Beta Functions**, **Dispersion**, **Alpha Functions**, **Orbit**, **Phase Advance**, **Beam Size**, **Lattice Summary**, **Lattice Diff**, **Beamline Bar**, **Floor Plan X-Z**, **Floor Plan Y-Z**

Once added, each panel appears at the top of the tab with controls to:

- Reorder it up or down in the stack
- Set its height in pixels
- Toggle element annotations
- Remove it

### Custom and Expression Panels

For more control, use **+ Custom panel** to select which Twiss parameters to display on a single set of axes, or **+ Expression panel** to define mathematical expressions (e.g. `beta_a * beta_b`) and plot derived quantities. Click **Browse data** inside the expression panel to see all available attributes for your backend.

### Panel Options

At the bottom of the Panels tab:

- **Show tune / chromaticity** — annotates Q, Q', Q'' on the first panel
- **Show panel titles** — toggles panel header labels
- **Beamline bar lite** — faster rendering for large lattices
- **Panel spacing** — sets the pixel gap between panels

---

## 5. Configure Beam Settings (optional)

If you are plotting **Beam Size**, switch to the **Beam Settings** tab and enter your beam parameters:

- **Emittance type** — Geometric or Normalized
- **Emit-x / Emit-y** — horizontal and vertical emittances in m·rad
- **Energy spread** (σ_dp) — relative momentum spread δp/p
- **n·σ** — multiplier for the beam size envelope (e.g. 1, 3)

---

## 6. Visual Customization (optional)

The **Visual** tab provides additional display controls:

- **Aspect ratio** — set a fixed W:H ratio for the plot
- **Floor plan scaling** — control element marker ratios and Y-range for X-Z and Y-Z projections
- **Display toggles** — dark mode, tunnel overlay, color beampipes, flip bends, legend placement
- **Font sizes** — override axis labels, tick labels, titles, annotations, and legend text independently

---

## 7. Run

Click **▶ Run**. The Output Log will show progress in real time. When the run completes, the **Open Plot** button activates — click it to open the interactive HTML file in your default browser.

If the browser does not open automatically, locate the output HTML file in your working directory and open it manually.

!!! tip "Dry Run"
    Use **Dry Run** to validate your configuration without generating the full plot. Useful for catching input errors quickly on large lattices.

---

## 8. Interacting with the Plot

The output is a fully interactive Plotly dashboard. All panels are linked — zooming on one panel updates the s-axis range across all panels simultaneously.

**What you can do:**

- **Pan and zoom** — click and drag, or use the scroll wheel
- **Hover** — hover over any curve or element marker to see its name, s-position, and value
- **Toggle traces** — click any legend entry to show or hide that trace; double-click to isolate it
- **Reset view** — double-click the plot area to reset zoom
- **Save PNG** — use the camera icon in the Plotly toolbar to export a static image directly from the browser

---

## 9. Export

For static image export or raw data, use the **Export** tab before running:

- **Save PNG / Save PDF** — generates a high-resolution static image (requires `kaleido`: `pip install kaleido`). 300 DPI is recommended for publication.
- **CSV export** — specify a base name (e.g. `lattice`) and RanOptics will write categorized CSV files such as `lattice-twiss.csv` and `lattice-orbit.csv`.

---

## 10. Comparing Lattices

To overlay or compare multiple lattices, use the **Compare Files** section in the Input tab:

- Click **+ Add file** to load one or more additional lattice files
- Set **Compare mode** to `Overlay` to plot all lattices on the same axes
- Enable **Normalize s (0→1)** to align lattices of different total lengths on a relative scale
