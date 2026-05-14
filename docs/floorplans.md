# Floor Plans

RanOptics supports two floor plan views: X-Z (horizontal) and Y-Z (vertical). Both show element positions in physical survey coordinates with color-coded element type blocks.

---

## Floor Plan X-Z

Shows the horizontal layout of the accelerator in the X-Z plane. This is the standard "bird's eye" view.

---

## Floor Plan Y-Z

Shows the vertical layout in the Y-Z plane. Useful for machines with significant vertical excursion (e.g. energy recovery linacs, recirculating machines).

### Ring machines — splitting by X side

For circular machines, the Y-Z floor plan shows both the front (+X) and back (−X) of the ring on the same panel, which creates visual overlap and clutter.

To separate them, add **two Floor Plan Y-Z panels** and use the half-selector button (labelled **½ Full**) on each panel row:

- First panel → click to **+X** (elements with positive X coordinate)
- Second panel → click to **-X** (elements with negative X coordinate)

This works correctly for multi-pass machines (e.g. CEBAF) since the split is based on the physical X coordinate of each element, not on pass number or arc angle.

---

## Tunnel wall overlay

If you have a tunnel wall coordinate file, specify it in the **Tunnel wall file** field. The file format is space or comma-separated columns:

```
x_in  y_in  z_in  x_out  y_out  z_out
```

One row per survey point. The inner and outer wall curves are drawn as overlays on the floor plan.

---

## Element colors

| Type | Color |
|------|-------|
| Dipole / SBend | Red |
| Quadrupole | Blue |
| Sextupole | Yellow |
| Kicker / Corrector | Orange |
| RF Cavity | Cyan |
| Monitor / BPM | Grey (hidden by default) |

Toggle **Show markers** in the Visual tab to show or hide monitors and BPMs.

---

## Floor plan in floor layout mode

Set **Layout mode** to `floor` in the Input tab to get a standalone floor plan figure without the optics panels. Useful for producing clean layout diagrams.
