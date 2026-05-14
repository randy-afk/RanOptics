# Expression Panels

Expression panels let you plot any user-defined Python expression using the lattice data loaded by RanOptics. This is the most flexible panel type — if the data exists in your lattice file, you can plot it.

---

## How it works

Type a Python expression in the Y1 (or Y2) field. RanOptics evaluates it against a namespace populated with all available lattice arrays and scalar parameters from the current backend.

Multiple expressions can be plotted on the same panel by separating them with commas:

```
beta_a, beta_b
```

---

## Available variables

The available variable names depend on your backend — RanOptics uses each backend's native naming conventions.

=== "ELEGANT"

    All columns from the `.twi` file are available directly, e.g.:

    ```
    betax, betay, etax, etay
    psix, psiy
    alphax, alphay
    dbetax/dp, dbetay/dp
    ```

    Scalar parameters from the `.twi` header are also available:

    ```
    nux, nuy
    dnux/dp, dnux/dp2, dnux/dp3
    dnuy/dp, dnuy/dp2, dnuy/dp3
    alphac, alphac2, alphac3
    ex0, Sdelta0, U0
    I1, I2, I3, I4, I5
    ```

    !!! note "Slash in names"
        Parameters like `dnux/dp` contain `/` which Python interprets as division. RanOptics automatically handles this — you can type `dnux/dp` directly and it will work.

=== "Tao / Bmad"

    Tao dot-notation is supported directly:

    ```
    beta.a, beta.b
    alpha.a, alpha.b
    eta.x, eta.y
    orbit.x, orbit.y
    phi.a, phi.b
    emit.a, emit.b
    ```

    RanOptics fetches these from the live Tao instance automatically.

=== "MAD-X"

    All TFS columns are available using their MAD-X names:

    ```
    BETX, BETY
    ALFX, ALFY
    DX, DY
    X, Y
    MUX, MUY
    ```

    TFS header scalars are also available:

    ```
    Q1, Q2
    DQ1, DQ2
    ALPHAC
    ```

=== "xsuite"

    xsuite twiss column names:

    ```
    betx, bety
    alfx, alfy
    dx, dy
    x, y
    mux, muy
    ```

    Scalar summary values (e.g. `qx`, `qy`, `dqx`, `dqy`) are also available.

---

## Math expressions

Standard Python math and NumPy are available:

```python
# Geometric mean of beta functions
sqrt(beta_a * beta_b)

# Normalized dispersion
etax / sqrt(betax)

# Custom combination
(beta_a - beta_b) / (beta_a + beta_b)
```

Available functions: `sqrt`, `abs`, `sin`, `cos`, `tan`, `exp`, `log`, `pi`, and all NumPy functions via `np`.

---

## Y1 and Y2 axes

Use the Y2 field to add a secondary right-hand axis. Useful when two quantities have very different scales:

- Y1: `betax, betay`
- Y2: `etax`

---

## Labels

Set **Y1 label** and **Y2 label** to override the auto-generated axis labels.

---

## Examples

```python
# Chromaticity contribution per element (ELEGANT)
dbetax/dp * kx

# Phase advance per unit length
psix / s

# Beam size from emittance (manual calculation)
sqrt(emit_x * betax + (etax * sigma_dp)**2) * 1e3
```
