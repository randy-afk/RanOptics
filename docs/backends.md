# Backends

RanOptics supports four simulation backends. The backend is auto-detected from the input file extension, but can be overridden manually.

---

## Tao / Bmad

**File extension:** `.init`

**Requires:** `pytao` Python package and the Bmad shared library.

RanOptics launches a live Tao instance in the background. This means:

- All Tao dot-notation attributes are available in expression panels (e.g. `beta.a`, `emit.a`)
- Multi-universe layouts are supported — each universe can be overlaid or compared
- Floor plan coordinates are read from Tao's `floor_position` data

**Multi-universe support:**

When your `.init` file defines multiple universes, RanOptics detects them automatically. Use the universe selector to choose which universes to plot and how to display them.

---

## ELEGANT

**File extension:** `.ele`

**Requires:** `elegant` and `sddsconvert` on your `PATH`.

RanOptics runs ELEGANT, then reads the `.twi`, `.cen`, and `.sig` output files. All columns and scalar header parameters from these files are available in expression panels.

**Floor plan:**

If a `.flr` file is found in the same directory, RanOptics reads it for floor plan coordinates.

**Beam parameters:**

Radiation integrals, emittances, and other global parameters from the `.twi` header are available as scalars in expression panels.

---

## xsuite

**File extension:** `.json`

**Requires:** `xsuite` Python package.

RanOptics loads the xsuite line from the JSON file and runs `line.twiss()`. The twiss table columns and scalar summary values are all available.

**Twiss method:**

You can select the twiss method (`4d` or `6d`) in the Beam Settings tab.

---

## MAD-X

**File extension:** `.tfs`

**Requires:** Nothing — no Python package needed.

Run MAD-X yourself to produce a `twiss.tfs` file (and optionally a `survey.tfs` for floor plan support), then point RanOptics at the twiss file.

**Survey / floor plan:**

If you have a `survey.tfs` file, specify it in the **MAD-X survey file** field in the Input tab. This enables the Floor Plan panels.

**TFS scalars:**

All parameters from the TFS header (`ALPHAC`, `Q1`, `Q2`, `DQ1`, `DQ2`, `ENERGY`, etc.) are available in expression panels.

---

## Auto-detection

| Extension | Backend |
|-----------|---------|
| `.init`   | Tao |
| `.ele`    | ELEGANT |
| `.json`   | xsuite |
| `.tfs`    | MAD-X |

Override by typing a different backend in the **Code backend** field.
