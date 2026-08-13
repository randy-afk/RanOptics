# =============================================================================
# core/utils.py — RanOptics shared utility functions and GUI helpers
# =============================================================================

from __future__ import annotations
import importlib.util, shutil
import numpy as np

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

import core.themes as _th

# Live accessors — always reads current theme values after apply_theme()
def _t(): return _th

# ── Backend availability ─────────────────────────────────────────────────────

def check_backend_ready(code):
    """Check whether the backend for `code` is actually usable before
    attempting to load a file with it.

    Returns None if ready, or a user-facing message describing what's
    missing. MAD-X needs no check — RanOptics only parses its output files
    and never invokes madx itself.
    """
    if code == 'tao':
        if importlib.util.find_spec('pytao') is None:
            return ("Tao/Bmad backend not found.\n\n"
                     "pytao is not installed (or its Bmad shared libraries "
                     "aren't on the library path). Install pytao and Bmad, "
                     "then restart RanOptics. See docs/installation.md.")
    elif code == 'elegant':
        missing = [exe for exe in ('elegant', 'sddsconvert') if shutil.which(exe) is None]
        if missing:
            return (f"ELEGANT backend not found: {', '.join(missing)} not on PATH.\n\n"
                     "Install elegant/sddsconvert and make sure they're on your "
                     "system PATH, then restart RanOptics. See docs/installation.md.")
    elif code == 'xsuite':
        if importlib.util.find_spec('xtrack') is None:
            return ("xsuite backend not found.\n\n"
                     "Install it with: pip install xsuite\n"
                     "See docs/installation.md.")
    return None

# ── Element visual helpers ────────────────────────────────────────────────────

# Default element colors — overridden by user via Colors tab
_ELEM_COLOR_DEFAULTS = {
    'sbend':      'red',
    'quadrupole': 'blue',
    'sextupole':  'green',
    'kicker':     'orange',
    'monitor':    'purple',
    'marker':     '#888888',
    'rfcavity':   'cyan',
    'lcavity':    'cyan',
}
# Fallback swatch color for element type keys with no entry above (e.g. 'other', 'drift')
_ELEM_COLOR_FALLBACK = '#888888'

def element_color(key, elem_colors=None):
    """Return the plot color for an element type.
    elem_colors: optional dict mapping type key (e.g. 'sbend') to hex/name color.
    """
    k = key.lower()
    ec = elem_colors or {}
    if 'sbend'      in k: return ec.get('sbend',      _ELEM_COLOR_DEFAULTS['sbend'])
    if 'quadrupole' in k: return ec.get('quadrupole', _ELEM_COLOR_DEFAULTS['quadrupole'])
    if 'sextupole'  in k: return ec.get('sextupole',  _ELEM_COLOR_DEFAULTS['sextupole'])
    if k in ('hkicker', 'vkicker', 'kicker'): return ec.get('kicker', _ELEM_COLOR_DEFAULTS['kicker'])
    if 'monitor'    in k: return ec.get('monitor',    _ELEM_COLOR_DEFAULTS['monitor'])
    if 'marker'     in k: return ec.get('marker',     _ELEM_COLOR_DEFAULTS['marker'])
    if 'rfcavity'   in k: return ec.get('rfcavity',   _ELEM_COLOR_DEFAULTS['rfcavity'])
    if 'lcavity'    in k: return ec.get('lcavity',    _ELEM_COLOR_DEFAULTS['lcavity'])
    return None

FULL_WIDTH_TYPES = ('sbend', 'quadrupole')
THIN_ELEMENT_THRESHOLD = 1e-3

def element_thickness(key, ft):
    return ft if any(t in key.lower() for t in FULL_WIDTH_TYPES) else ft / 2.0

def element_polygon(x0, y0, theta0, L, angle, thickness):
    t = thickness / 2.0
    if abs(angle) < 1e-6:
        dx, dy = np.cos(theta0) * L, np.sin(theta0) * L
        nx, ny = -np.sin(theta0) * t, np.cos(theta0) * t
        return ([x0+nx, x0+dx+nx, x0+dx-nx, x0-nx, x0+nx],
                [y0+ny, y0+dy+ny, y0+dy-ny, y0-ny, y0+ny])
    rho = L / angle; n_pts = max(30, int(abs(angle) * 80))
    angs = np.linspace(0, angle, n_pts)
    cx = x0 - rho * np.sin(theta0); cy = y0 + rho * np.cos(theta0)
    ox = cx + (rho + t) * np.sin(theta0 + angs)
    oy = cy - (rho + t) * np.cos(theta0 + angs)
    ix = cx + (rho - t) * np.sin(theta0 + angs)
    iy = cy - (rho - t) * np.cos(theta0 + angs)
    return (np.concatenate([ox, ix[::-1], [ox[0]]]).tolist(),
            np.concatenate([oy, iy[::-1], [oy[0]]]).tolist())

def element_oval(x0, y0, theta0, L, thickness):
    a, b = L / 2.0, thickness / 2.0
    t = np.linspace(0, 2 * np.pi, 49)
    u, v = a * np.cos(t), b * np.sin(t)
    cx = x0 + (L / 2) * np.cos(theta0); cy = y0 + (L / 2) * np.sin(theta0)
    return (cx + u * np.cos(theta0) - v * np.sin(theta0)).tolist(), \
           (cy + u * np.sin(theta0) + v * np.cos(theta0)).tolist()

def make_hover(elem):
    name = elem['name'].split('\\')[-1]; key = elem['key']; L = elem['length']
    k1 = elem.get('k1', 0.0); k2 = elem.get('k2', 0.0)
    angle = elem.get('angle', 0.0)
    raw_angle = elem.get('raw_angle', angle)
    s0 = elem['s_start']; kc = key.lower()
    lines = [f'<b>{name}</b>', f'<i>{key}</i>',
             f'L = {L:.4f} m',
             f's_start = {s0:.4f} m',
             f's_end &nbsp;= {s0+L:.4f} m']
    if 'sbend' in kc:
        rt = elem.get('ref_tilt', 0.0)
        bend_plane = 'Vertical' if abs(abs(rt) - np.pi / 2) < 0.01 else 'Horizontal'
        lines.append(f'Bend plane: {bend_plane}')
        lines.append(f'Angle = {np.degrees(raw_angle):.4f}°')
        if abs(raw_angle) > 1e-9: lines.append(f'ρ = {abs(L/raw_angle):.4f} m')
    if 'quadrupole' in kc: lines.append(f'K1 = {k1:.6f} m⁻²')
    if 'sextupole'  in kc: lines.append(f'K2 = {k2:.6f} m⁻³')
    if kc == 'kicker':
        lines += [f'hkick={elem.get("hkick",0):.6f}', f'vkick={elem.get("vkick",0):.6f}']
    elif kc in ('hkicker', 'vkicker'):
        lines.append(f'kick={elem.get("kick",0):.6f}')
    if 'rfcavity' in kc or 'lcavity' in kc:
        v = elem.get('voltage', 0.0); f = elem.get('frequency', 0.0)
        if v: lines.append(f'V={v/1e6:.3f} MV' if abs(v) >= 1e6 else f'V={v:.1f} V')
        if f: lines.append(f'f={f/1e9:.4f} GHz' if f >= 1e9 else f'f={f/1e6:.4f} MHz')
    return '<br>'.join(lines) + '<extra></extra>'

def panel_title(p):
    return {
        'twiss':      'Beta Functions & Dispersion',
        'phase':      'Phase Advance',
        'orbit':      'Orbit',
        'beamsize':   'Beam Size',
        'beta':       'Beta Functions',
        'dispersion': 'Dispersion',
        'alpha':      'Alpha Functions',
        'summary':    'Lattice Summary',
        'latdiff':    'Lattice Diff',
        'bar':        'Beamline',
        'floor-xz':   'Floor Plan — X vs Z',
        'floor-yz':   'Floor Plan — Y vs Z',
    }.get(p, '')

# ── Parse helpers ─────────────────────────────────────────────────────────────

def _parse_fp_range(rng_str):
    """Parse a range string like '-0.5:0.5' into [float, float] or None."""
    if not rng_str: return None
    try:
        parts = str(rng_str).split(':')
        if len(parts) != 2: return None
        return [float(parts[0].strip()), float(parts[1].strip())]
    except (ValueError, AttributeError):
        return None

def _parse_yrange(text):
    """Parse a 'min:max' string into [float, float] or '' on failure."""
    if not text or ':' not in text:
        return ''
    try:
        parts = text.split(':', 1)
        return [float(parts[0].strip()), float(parts[1].strip())]
    except (ValueError, AttributeError):
        return ''

# ── Log line classifier ───────────────────────────────────────────────────────

def _clf(line):
    lo = line.lower()
    if any(w in lo for w in ("error", "traceback", "exception", "failed", "✗")):
        return "error"
    if any(w in lo for w in ("warning", "warn")):
        return "warn"
    if any(w in lo for w in ("saved", "done", "complete", "✓")):
        return "ok"
    return "info"

# ── GUI helper widgets ────────────────────────────────────────────────────────

def _make_scroll_widget(parent=None):
    """Returns (scroll_area, inner_widget, vbox_layout)."""
    sa = QScrollArea(parent)
    sa.setWidgetResizable(True)
    sa.setStyleSheet(_th._SCROLL_SS)
    inner = QWidget()
    inner.setStyleSheet("background: transparent;")
    vbox = QVBoxLayout(inner)
    vbox.setContentsMargins(0, 4, 0, 8)
    vbox.setSpacing(0)
    sa.setWidget(inner)
    return sa, inner, vbox

def _sec(layout, title):
    """Section header: uppercase faint label + horizontal rule."""
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(8, 12, 8, 4); h.setSpacing(8)
    lbl = QLabel(title.upper()); lbl.setFont(_th.FONT_SMALL)
    lbl.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent; font-weight: 600; letter-spacing: 1.6px;")
    line = QFrame(); line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background: {_th.BORDER}; border: none;")
    line.setFixedHeight(1)
    h.addWidget(lbl); h.addWidget(line, 1)
    layout.addWidget(w)

def _card(layout):
    """Transparent container widget. Returns (widget, vbox)."""
    w = QWidget(); w.setStyleSheet("background: transparent;")
    v = QVBoxLayout(w); v.setContentsMargins(0, 2, 0, 2); v.setSpacing(0)
    layout.addWidget(w); return w, v

def _row(layout):
    """Horizontal row. Returns QHBoxLayout added to layout."""
    w = QWidget(); w.setStyleSheet("background: transparent;")
    h = QHBoxLayout(w); h.setContentsMargins(8, 2, 8, 2); h.setSpacing(6)
    h.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.addWidget(w)
    return h

def _lbl(layout, text, width=160):
    lbl = QLabel(text); lbl.setFont(_th.FONT_MAIN)
    lbl.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent;")
    lbl.setFixedWidth(width); lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.addWidget(lbl)

def _ent(layout, width=200, placeholder=""):
    e = QLineEdit(); e.setFont(_th.FONT_MONO)
    e.setPlaceholderText(placeholder); e.setFixedWidth(width)
    e.setStyleSheet(_th._ENTRY_SS)
    layout.addWidget(e); return e

def _btn(layout, text, cmd, width=80, color=None):
    if color is None: color = _th.ACCENT
    b = QPushButton(text); b.setFont(_th.FONT_MAIN); b.setFixedWidth(width)
    b.clicked.connect(cmd)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {_th.PANEL}; border: 1px solid {_th.BORDER};
            border-radius: 8px; color: {color}; padding: 4px 8px;
        }}
        QPushButton:hover {{ background: {_th.SURFACE2}; border-color: {color}; }}
        QPushButton:pressed {{ background: {_th.BORDER}; }}
        QPushButton:disabled {{ color: {_th.FG_DIM}; border-color: {_th.BORDER}; background: {_th.PANEL}; }}
    """)
    layout.addWidget(b); return b

def _chk(layout, text):
    c = QCheckBox(text); c.setFont(_th.FONT_MAIN); c.setStyleSheet(_th._CHK_SS)
    layout.addWidget(c); return c

def _dd(layout, items, width=120):
    cb = QComboBox(); cb.setFont(_th.FONT_MAIN); cb.addItems(items)
    cb.setFixedWidth(width); cb.setStyleSheet(_th._COMBO_SS)
    layout.addWidget(cb); return cb

def _hint(layout, text):
    lbl = QLabel(text); lbl.setFont(_th.FONT_SMALL)
    lbl.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent;")
    layout.addWidget(lbl)

def _help(layout, text):
    """Dimmed description line."""
    lbl = QLabel(text); lbl.setFont(_th.FONT_SMALL); lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent; padding: 0 12px 4px 12px;")
    layout.addWidget(lbl)

def _rb(layout, text, group, val, cmd=None):
    r = QRadioButton(text); r.setFont(_th.FONT_MAIN); r.setStyleSheet(_th._RB_SS)
    r.setProperty("value", val)
    if cmd: r.clicked.connect(cmd)
    group.addButton(r); layout.addWidget(r); return r
