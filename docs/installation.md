# Installation

## Requirements

- Python 3.10 or later
- One or more supported simulation backends (see below)

---

## Install with conda (recommended)

```bash
git clone https://github.com/randy-afk/RanOptics.git
cd RanOptics
conda env create -f environment.yml
conda activate ranoptics
```

---

## Install with pip

```bash
git clone https://github.com/randy-afk/RanOptics.git
cd RanOptics
pip install -r requirements.txt
```

---

## Backend dependencies

=== "Tao / Bmad"

    Install `pytao` via pip:

    ```bash
    pip install pytao
    ```

    See the [pytao documentation](https://github.com/bmad-sim/pytao) for full installation instructions including the Bmad shared library.

=== "ELEGANT"

    `elegant` and `sddsconvert` must be on your `PATH`. No Python package is needed.

    To verify:
    ```bash
    which elegant
    which sddsconvert
    ```

=== "xsuite"

    ```bash
    pip install xsuite
    ```

    See the [xsuite documentation](https://xsuite.readthedocs.io) for details.

=== "MAD-X"

    No Python package needed. Run MAD-X yourself to produce `twiss.tfs` and optionally `survey.tfs`, then point RanOptics at those files.

---

## Optional: PNG/PDF export

PNG and PDF export requires `kaleido`:

```bash
pip install kaleido
```

---

## Running RanOptics

```bash
python RanOptics_new.py
```
