# =============================================================================
# tests/fixtures.py — synthetic lattice used by tests/test_plot_optics.py
#
# A minimal closed ring built purely in Python via xtrack — no external
# lattice file needed. Only good for exercising plot_optics()'s layout and
# rendering code paths; not a physically meaningful machine.
# =============================================================================

from __future__ import annotations
import math


def build_ring_json(path):
    """Build a small closed FODO-style ring and save it as an xsuite JSON
    lattice at `path`. Requires xtrack."""
    import xtrack as xt

    n_cells = 4
    bend_angle = (2 * math.pi) / n_cells / 2  # 2 bends per cell -> closes at 2*pi

    names, elements = [], []
    def add(name, el):
        names.append(name)
        elements.append(el)

    for i in range(n_cells):
        add(f'qf{i}', xt.Quadrupole(length=0.3, k1=0.35))
        add(f'd{i}a', xt.Drift(length=0.5))
        add(f'b{i}a', xt.Bend(length=1.0, angle=bend_angle, k0=bend_angle))
        add(f'd{i}b', xt.Drift(length=0.5))
        add(f'qd{i}', xt.Quadrupole(length=0.3, k1=-0.35))
        add(f'd{i}c', xt.Drift(length=0.5))
        add(f'sf{i}', xt.Sextupole(length=0.2, k2=0.4))
        add(f'd{i}d', xt.Drift(length=0.5))
        add(f'b{i}b', xt.Bend(length=1.0, angle=bend_angle, k0=bend_angle))
        add(f'd{i}e', xt.Drift(length=0.5))

    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(p0c=6500e9, q0=1, mass0=xt.PROTON_MASS_EV)
    line.build_tracker()
    line.twiss(method='4d')  # confirms a periodic solution exists before saving
    line.to_json(str(path))
