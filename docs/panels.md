# Panels

Panels are the building blocks of a RanOptics plot. Stack them in any order, control their height individually, and add element annotations per panel.

---

## Preset panels

### Twiss & Dispersion
Shows βₓ, βᵧ on the left axis and ηₓ, ηᵧ on the right axis. The most commonly used panel.

### Beta Functions
βₓ and βᵧ only, without dispersion.

### Dispersion
ηₓ and ηᵧ only.

### Alpha Functions
αₓ and αᵧ.

### Orbit
Closed orbit x and y positions along the lattice.

### Phase Advance
Accumulated phase advance μₓ and μᵧ.

### Beam Size
σₓ and σᵧ computed from emittance and dispersion. Configure emittance and beam energy in the **Beam Settings** tab.

### Floor Plan X-Z
Horizontal floor plan showing element positions in the X-Z plane. Elements are color-coded by type.

### Floor Plan Y-Z
Vertical floor plan showing element positions in the Y-Z plane.

!!! tip "Ring machines"
    For ring machines, use two Floor Plan Y-Z panels. Set one to **+X** and one to **-X** using the half-selector button on the panel row. This splits the ring into its two visible sides without the overlapping element mess.

### Beamline Bar
A compact element layout bar showing element types along s. Supports wildcard element name annotations (e.g. `IPM*`, `QF*,QD*`).

!!! tip "Large lattices"
    Enable **Beamline bar lite** in Panel Options for significantly faster rendering on large lattices.

### Lattice Summary
A table showing tune, chromaticity, radiation integrals, and element counts. One row per universe (Tao) or per file (compare mode).

### Lattice Diff
Element-by-element comparison between the primary lattice and a compare file. Shows differences in strengths, entry positions, and exit positions.

---

## Custom panels

Click **+ Custom panel...** to open the panel builder. Select any combination of standard quantities on Y1 and Y2 axes. Give it a name — this name also appears as the Y-axis label.

---

## Expression panels

Click **+ Expression panel...** to define your own Python expression using lattice data. See [Expression Panels](expressions.md) for full details.

---

## Panel controls

Each panel row in the Panels tab has:

| Control | Description |
|---------|-------------|
| Height field | Panel height in pixels |
| ▲ ▼ | Move panel up or down |
| × | Remove panel |
| ⋮ (annotation) | Wildcard pattern for element name annotations, e.g. `IPM*` |
| ÷ (legend) | Override legend position for this panel |
| ½ (Y-Z only) | Split ring: Full / +X side / -X side |
