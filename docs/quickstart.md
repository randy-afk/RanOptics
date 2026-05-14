# Quickstart

This guide walks you through plotting your first lattice with RanOptics.

---

## 1. Launch

```bash
python RanOptics_new.py
```

---

## 2. Select your input file

Click **Browse** next to *Input file* and select your simulation file.

| Extension | Backend |
|-----------|---------|
| `.init`   | Tao / Bmad |
| `.ele`    | ELEGANT |
| `.json`   | xsuite |
| `.tfs`    | MAD-X |

The backend is auto-detected. You can override it manually in the *Code backend* field.

---

## 3. Configure panels

In the **Panels** tab, select which plots to include. The default layout includes:

- Floor Plan X-Z
- Twiss & Dispersion
- Beamline Bar

Click any preset button to add a panel. Use ▲▼ to reorder, × to remove.

The number field next to each panel controls its pixel height.

---

## 4. Run

Click **▶ Run**. Watch the output log for progress.

When the run completes, **Open Plot** lights up green. Click it to open your plot in the browser.

---

## 5. Export

- **HTML** — the output file is saved automatically to the path shown in *Output HTML*
- **CSV** — click **Export CSV** to save one CSV file per panel
- **PNG/PDF** — configure in the **Export** tab (requires `kaleido`)

---

## Tips

- Use **Dry Run** to validate your input file without running the full simulation
- Use **Presets** → **Save** to save your current configuration for reuse
- The **Range START:END** field accepts element names (`QUA01:QUA06`) or s positions (`3.0:19.0`)
