# Compare Mode

RanOptics can compare multiple lattice files against a primary file. This is useful for comparing design vs. measured optics, different tunes, or different lattice versions.

---

## Adding compare files

In the **Input** tab, click **+ Add file** under *Compare Files* to add one or more additional files. Each file must use the same backend as the primary.

---

## Compare modes

| Mode | Description |
|------|-------------|
| **Overlay** | All files plotted on the same axes with different colors |
| **Separate** | Each file gets its own set of panels, stacked vertically |
| **Difference** | Primary minus compare file, in physical units |
| **Difference (%)** | Relative difference as a percentage |

---

## Normalize s

Enable **Normalize s (0→1)** to map the s coordinate of all files to [0, 1]. This aligns lattices of different total lengths for meaningful comparison.

---

## Multi-universe (Tao only)

For Tao lattices with multiple universes defined in the `.init` file, RanOptics detects them automatically. Each universe can be treated as a separate dataset for comparison within a single file.
