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

### Pointing RanOptics at your Bmad library

Two extra fields appear when Tao is selected: **Bmad library** and **Extra library dirs**.

You only need them in the **standalone executable**. Running from source, with `pytao` and Bmad already on your environment, both can stay blank: Python finds the library the same way it always does. The packaged executable is isolated from your Python environment, so it cannot auto-discover an installed Bmad and has to be told where it is.

#### Finding `libtao.so`

If Bmad came from conda-forge (the usual case), it sits in your environment's `lib` directory. With that environment **activated**:

```bash
ls "$CONDA_PREFIX"/lib/libtao.so
```

!!! warning "Activate the environment first"
    `$CONDA_PREFIX` is empty when no environment is active, so the command above quietly checks `/lib/libtao.so` and reports "No such file" even when Bmad is installed. If you see that, run `conda activate` first, or name the path explicitly:

    ```bash
    ls ~/miniforge3/envs/YOUR_ENV/lib/libtao.so
    ```

Still not found, search the likely roots:

```bash
find ~ /opt /usr/local -name 'libtao.so*' 2>/dev/null
```

Or ask the dynamic linker, if the library is on the system path:

```bash
ldconfig -p | grep libtao
```

On macOS look for `libtao.dylib`, on Windows `libtao.dll`. Paste the full path into **Bmad library**.

#### When you also need Extra library dirs

Leave this blank first and try a run. It is only needed when libtao's own dependencies (GSL, LAPACK, FFTW3, HDF5 and friends) are *not* sitting next to `libtao.so`, which happens with a hand-built Bmad or one assembled from system packages rather than a single conda environment.

To see whether anything is missing, ask what libtao can't resolve:

```bash
ldd /path/to/libtao.so | grep 'not found'
```

Every line printed is a dependency RanOptics can't reach. Locate each one and add its directory to **Extra library dirs**, comma-separated:

```bash
find / -name 'libgsl.so*' 2>/dev/null
```

A clean conda install prints nothing from the `ldd` check, which means the field can stay empty.

#### Both fields are remembered

They are saved to `~/.ranoptics_settings.json` as soon as you finish editing them, and restored automatically the next time the app starts. You enter them once, not every launch. They are deliberately kept out of presets, since a path into your local Bmad install means nothing on another machine.

#### What happens under the hood

RanOptics reads `libtao.so`'s dependency list, follows it recursively, and gathers the library plus everything it genuinely needs into a single directory under your system temp folder, named `ranoptics_bmad_<id>`. This is what makes a scattered install work: the loader resolves a library's dependencies from its own directory, so putting them together is what lets it find them.

That directory is reused across runs and sessions, and rebuilt automatically if you move or upgrade your Bmad install. Deleting it is harmless, it is recreated on the next run.

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
