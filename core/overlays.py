# =============================================================================
# core/overlays.py — RanOptics panel overlays and data browsers
# =============================================================================

from __future__ import annotations
import os, re, subprocess, sys, threading
from pathlib import Path
import numpy as np

from PySide6.QtCore    import Qt, Signal, QObject
from PySide6.QtGui     import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget,
)

from core.themes import (
    ACCENT, ACCENT2, BG, BORDER, CRUST, ERROR, FG, FG_DIM, FG_LBL,
    MANTLE, PANEL, SUCCESS, SURFACE2, HIGHLIGHT, WARN,
    FONT_BOLD, FONT_MAIN, FONT_MONO, FONT_SEC, FONT_SMALL,
    _BTN_SS, _CHK_SS, _COMBO_SS, _ENTRY_SS, _RB_SS, _SCROLL_SS,
)
from core.utils import _make_scroll_widget, _help, _sec
from core.engine import _inspect_available_data
from core.loaders import _read_tfs, load_xsuite

# ── Data browser constants ────────────────────────────────────────────────────
_TAO_DATA_CATEGORIES = {
    "Twiss — Normal Mode": [
        ("alpha.a",    "Normal-mode alpha function"),
        ("alpha.b",    "Normal-mode alpha function"),
        ("beta.a",     "Normal-mode beta function (m)"),
        ("beta.b",     "Normal-mode beta function (m)"),
        ("beta.c",     "Normal-mode beta function (m)"),
        ("gamma.a",    "Normal-mode gamma function (1/m)"),
        ("gamma.b",    "Normal-mode gamma function (1/m)"),
        ("emit.a",     "Normal-mode emittance (m·rad)"),
        ("emit.b",     "Normal-mode emittance (m·rad)"),
        ("emit.c",     "Normal-mode emittance (m·rad)"),
        ("eta.a",      "Normal-mode dispersion (m)"),
        ("eta.b",      "Normal-mode dispersion (m)"),
        ("etap.a",     "Normal-mode momentum dispersion"),
        ("etap.b",     "Normal-mode momentum dispersion"),
    ],
    "Dispersion": [
        ("eta.x",      "Horizontal dispersion (m)"),
        ("eta.y",      "Vertical dispersion (m)"),
        ("etap.x",     "Horizontal momentum dispersion"),
        ("etap.y",     "Vertical momentum dispersion"),
        ("deta_ds.a",  "Dispersion derivative a-mode"),
        ("deta_ds.b",  "Dispersion derivative b-mode"),
        ("deta_ds.x",  "Horizontal dispersion derivative"),
        ("deta_ds.y",  "Vertical dispersion derivative"),
    ],
    "Orbit": [
        ("orbit.x",    "Horizontal phase-space orbit (m)"),
        ("orbit.y",    "Vertical phase-space orbit (m)"),
        ("orbit.z",    "Longitudinal phase-space orbit (m)"),
        ("orbit.px",   "Horizontal canonical momentum"),
        ("orbit.py",   "Vertical canonical momentum"),
        ("orbit.pz",   "Longitudinal canonical momentum"),
    ],
    "Phase Advance": [
        ("phi.a",      "Phase advance a-mode (2π units)"),
        ("phi.b",      "Phase advance b-mode (2π units)"),
    ],
    "Courant-Snyder": [
        ("alpha_a",    "Courant-Snyder alpha x"),
        ("alpha_b",    "Courant-Snyder alpha y"),
        ("beta_a",     "Courant-Snyder beta x (m)"),
        ("beta_b",     "Courant-Snyder beta y (m)"),
        ("gamma_a",    "Courant-Snyder gamma x (1/m)"),
        ("gamma_b",    "Courant-Snyder gamma y (1/m)"),
    ],
    "Beam Size / Emittance": [
        ("beam_energy",     "Beam energy (eV)"),
        ("e_tot",           "Total energy (eV)"),
        ("p0c",             "Reference momentum × c (eV)"),
        ("s",               "Longitudinal position (m)"),
        ("ref_time",        "Reference time (s)"),
    ],
    "W Function / Chromatic": [
        ("chrom.w.a",       "W function a-mode"),
        ("chrom.w.b",       "W function b-mode"),
        ("chrom.dw.a",      "dW/ds a-mode"),
        ("chrom.dw.b",      "dW/ds b-mode"),
    ],
    "Element Attributes": [
        ("k1",         "Quadrupole strength (1/m²)"),
        ("k2",         "Sextupole strength (1/m³)"),
        ("k3",         "Octupole strength (1/m⁴)"),
        ("angle",      "Bend angle (rad)"),
        ("l",          "Element length (m)"),
        ("tilt",       "Tilt angle (rad)"),
        ("x_offset",   "Horizontal offset (m)"),
        ("y_offset",   "Vertical offset (m)"),
        ("voltage",    "RF voltage (V)"),
        ("phi0",       "RF phase (rad)"),
    ],
}

# ── ELEGANT data browser constants ────────────────────────────────────────────
_ELEGANT_TWI_COLUMNS = [
    ("s",           "Longitudinal position (m)"),
    ("betax",       "Horizontal beta function (m)"),
    ("betay",       "Vertical beta function (m)"),
    ("alphax",      "Horizontal alpha function"),
    ("alphay",      "Vertical alpha function"),
    ("etax",        "Horizontal dispersion (m)"),
    ("etay",        "Vertical dispersion (m)"),
    ("etaxp",       "Horizontal dispersion prime"),
    ("etayp",       "Vertical dispersion prime"),
    ("psix",        "Horizontal phase advance (rad)"),
    ("psiy",        "Vertical phase advance (rad)"),
    ("x",           "Centroid x (m)"),
    ("y",           "Centroid y (m)"),
    ("xp",          "Centroid x' (rad)"),
    ("yp",          "Centroid y' (rad)"),
    ("Sx",          "RMS beam size x (m)"),
    ("Sy",          "RMS beam size y (m)"),
]

_ELEGANT_CEN_COLUMNS = [
    ("s",    "Longitudinal position (m)"),
    ("x",    "Centroid x (m)"),
    ("xp",   "Centroid x' (rad)"),
    ("y",    "Centroid y (m)"),
    ("yp",   "Centroid y' (rad)"),
    ("t",    "Time (s)"),
    ("p",    "Momentum deviation"),
]

_ELEGANT_SIG_COLUMNS = [
    ("s",    "Longitudinal position (m)"),
    ("Sx",   "RMS x (m)"),
    ("Sy",   "RMS y (m)"),
    ("Ss",   "RMS s (m)"),
    ("Sxp",  "RMS x' (rad)"),
    ("Syp",  "RMS y' (rad)"),
    ("Sdelta", "RMS δp/p"),
    ("ex",   "Emittance x (m·rad)"),
    ("ey",   "Emittance y (m·rad)"),
]

_ELEGANT_TWI_SCALARS = [
    ("pCentral",   "Central momentum (m_e c)"),
    ("Ex",         "Horizontal emittance (m·rad)"),
    ("Ey",         "Vertical emittance (m·rad)"),
    ("Sdelta0",    "Energy spread δp/p"),
    ("nux",        "Horizontal tune"),
    ("nuy",        "Vertical tune"),
    ("dnux/dp",    "Horizontal chromaticity"),
    ("dnuy/dp",    "Vertical chromaticity"),
]

class CustomPanelOverlay:
    """Custom panel composer built inside the Panels tab."""
    DTYPES = [
        ('Beta', 'beta'), ('Dispersion', 'disp'), ('Alpha', 'alpha'),
        ('Orbit', 'orbit'), ('Phase Advance', 'phase'), ('Beam Size', 'beamsize'),
    ]

    def __init__(self, layout, on_done):
        self._on_done = on_done
        self._y1_rows = []; self._y2_rows = []

        # Title row
        title_w = QWidget(); title_w.setStyleSheet("background: transparent;")
        title_h = QHBoxLayout(title_w); title_h.setContentsMargins(12, 12, 8, 4)
        lbl = QLabel("CUSTOM PANEL"); lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        title_h.addWidget(lbl); title_h.addStretch()
        add_b = QPushButton("Add"); add_b.setFont(FONT_BOLD); add_b.setFixedSize(80, 28)
        add_b.setStyleSheet(f"QPushButton {{ background: {ACCENT}; border-radius: 6px; color: white; }} QPushButton:hover {{ background: #3a9fff; }}")
        add_b.clicked.connect(self._ok)
        can_b = QPushButton("Cancel"); can_b.setFont(FONT_MAIN); can_b.setFixedSize(90, 28)
        can_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG_DIM}; }} QPushButton:hover {{ background: {BORDER}; }}")
        can_b.clicked.connect(lambda: on_done(None))
        title_h.addWidget(add_b); title_h.addWidget(can_b)
        layout.addWidget(title_w)

        # Scroll area
        sa, inner_w, inner_v = _make_scroll_widget()
        inner_w.setStyleSheet(f"background: #28282a;")
        layout.addWidget(sa)
        self._inner_v = inner_v
        self._inner_w = inner_w

        # Panel name
        lbl2 = QLabel("Panel name"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl2)
        self.e_name = QLineEdit("Custom Panel"); self.e_name.setFont(FONT_MONO)
        self.e_name.setFixedWidth(300); self.e_name.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_name)

        # Y1
        lbl_y1 = QLabel("Y1 AXIS  (left) — required"); lbl_y1.setFont(FONT_SEC)
        lbl_y1.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 6px 4px 2px 4px;")
        inner_v.addWidget(lbl_y1)
        self._y1_box_w = QWidget(); self._y1_box_w.setStyleSheet(f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;")
        self._y1_box_v = QVBoxLayout(self._y1_box_w); self._y1_box_v.setContentsMargins(4, 4, 4, 4)
        inner_v.addWidget(self._y1_box_w)
        self._add_y1_row()
        add_y1 = QPushButton("+ add dataset to Y1"); add_y1.setFont(FONT_MAIN)
        add_y1.setStyleSheet(f"QPushButton {{ background: transparent; color: {ACCENT2}; border: none; text-align: left; padding: 2px 4px; }} QPushButton:hover {{ color: white; }}")
        add_y1.clicked.connect(lambda: self._add_y1_row())
        inner_v.addWidget(add_y1)

        # Y2
        lbl_y2 = QLabel("Y2 AXIS  (right) — optional"); lbl_y2.setFont(FONT_SEC)
        lbl_y2.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 6px 4px 2px 4px;")
        inner_v.addWidget(lbl_y2)
        self._y2_box_w = QWidget(); self._y2_box_w.setStyleSheet(f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;")
        self._y2_box_v = QVBoxLayout(self._y2_box_w); self._y2_box_v.setContentsMargins(4, 4, 4, 4)
        inner_v.addWidget(self._y2_box_w)
        self._y2_box_w.hide()
        add_y2 = QPushButton("+ add dataset to Y2"); add_y2.setFont(FONT_MAIN)
        add_y2.setStyleSheet(add_y1.styleSheet())
        add_y2.clicked.connect(lambda: self._add_y2_row())
        inner_v.addWidget(add_y2)
        inner_v.addStretch()

    def _make_row(self, box_layout):
        outer = QWidget(); outer.setStyleSheet("background: transparent;")
        ov = QVBoxLayout(outer); ov.setContentsMargins(8, 6, 8, 6); ov.setSpacing(6)
        # Enable + X/Y checkboxes
        top = QWidget(); top.setStyleSheet("background: transparent;")
        th = QHBoxLayout(top); th.setContentsMargins(0, 0, 0, 0); th.setSpacing(16)
        en = QCheckBox("Enable"); en.setChecked(True); en.setFont(FONT_MAIN); en.setStyleSheet(_CHK_SS)
        xc = QCheckBox("X"); xc.setChecked(True); xc.setFont(FONT_MAIN); xc.setStyleSheet(_CHK_SS)
        yc = QCheckBox("Y"); yc.setChecked(True); yc.setFont(FONT_MAIN); yc.setStyleSheet(_CHK_SS)
        th.addWidget(en); th.addWidget(xc); th.addWidget(yc); th.addStretch()
        ov.addWidget(top)
        # Radio buttons for dtype (2 rows × 3)
        rg = QWidget(); rg.setStyleSheet("background: transparent;")
        rgg = QGridLayout(rg); rgg.setContentsMargins(0, 0, 0, 0); rgg.setSpacing(4)
        bg = QButtonGroup(rg)
        for i, (lbl_text, key) in enumerate(self.DTYPES):
            rb = QRadioButton(lbl_text); rb.setFont(FONT_MAIN); rb.setStyleSheet(_RB_SS)
            rb.setProperty("value", key)
            if i == 0: rb.setChecked(True)
            bg.addButton(rb); rgg.addWidget(rb, i // 3, i % 3)
        ov.addWidget(rg)
        box_layout.addWidget(outer)

        def _get():
            if not en.isChecked(): return []
            key = next((b.property("value") for b in bg.buttons() if b.isChecked()), 'beta')
            r = []
            if xc.isChecked(): r.append((key, 'x'))
            if yc.isChecked(): r.append((key, 'y'))
            return r
        return _get

    def _add_y1_row(self):
        if len(self._y1_rows) >= 4: return
        self._y1_rows.append(self._make_row(self._y1_box_v))
        self._y1_box_w.adjustSize()
        self._inner_w.adjustSize()

    def _add_y2_row(self):
        if len(self._y2_rows) >= 4: return
        if len(self._y2_rows) == 0: self._y2_box_w.show()
        self._y2_rows.append(self._make_row(self._y2_box_v))
        self._y2_box_w.adjustSize()
        self._inner_w.adjustSize()

    def _ok(self):
        y1 = [ds for fn in self._y1_rows for ds in fn()]
        if not y1:
            QMessageBox.warning(None, "Y1 Required", "Please enable at least one Y1 dataset.")
            return
        y2 = [ds for fn in self._y2_rows for ds in fn()]
        pname = self.e_name.text().strip() or "Custom Panel"
        self._on_done({'name': pname, 'y1': y1, 'y2': y2})

class ExprPanelOverlay:
    """Composer for expression-based panels — in-tab, no Toplevel."""

    def __init__(self, layout, on_done, code='tao',
                 input_file='', xsuite_twiss='4d', xsuite_line=None,
                 madx_survey=None):
        self._on_done      = on_done
        self._code         = code
        self._input_file   = input_file
        self._xsuite_twiss = xsuite_twiss
        self._xsuite_line  = xsuite_line
        self._madx_survey  = madx_survey
        self._last_entry   = None
        self._tao_browser      = None
        self._elegant_browser  = None
        self._madx_browser     = None
        self._xsuite_browser   = None

        # Title row
        title_w = QWidget(); title_w.setStyleSheet("background: transparent;")
        title_h = QHBoxLayout(title_w); title_h.setContentsMargins(12, 12, 8, 4)
        lbl = QLabel("EXPRESSION PANEL"); lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        title_h.addWidget(lbl); title_h.addStretch()
        add_b = QPushButton("Add"); add_b.setFont(FONT_BOLD); add_b.setFixedSize(80, 28)
        add_b.setStyleSheet(f"QPushButton {{ background: {ACCENT}; border-radius: 6px; color: white; }} QPushButton:hover {{ background: #3a9fff; }}")
        add_b.clicked.connect(self._ok)
        can_b = QPushButton("Cancel"); can_b.setFont(FONT_MAIN); can_b.setFixedSize(90, 28)
        can_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG_DIM}; }} QPushButton:hover {{ background: {BORDER}; }}")
        can_b.clicked.connect(lambda: on_done(None))
        title_h.addWidget(add_b); title_h.addWidget(can_b)
        layout.addWidget(title_w)

        sa, inner_w, inner_v = _make_scroll_widget()
        inner_w.setStyleSheet(f"background: #28282a;")
        layout.addWidget(sa)
        self._inner_v = inner_v
        self._inner_w = inner_w

        # Panel name
        lbl2 = QLabel("Panel name"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl2)
        self.e_name = QLineEdit("Expression Panel"); self.e_name.setFont(FONT_MONO)
        self.e_name.setFixedWidth(380); self.e_name.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_name)

        # Hint
        hint = {
            'tao':     "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + any Tao attribute + global scalars",
            'elegant': "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + twi columns + scalar parameters",
            'xsuite':  "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + twiss columns + global scalars",
        }.get(code, "")
        hl = QLabel(hint); hl.setFont(FONT_SMALL); hl.setWordWrap(True)
        hl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 4px 4px 4px;")
        inner_v.addWidget(hl)

        # Buttons row
        btn_row_w = QWidget(); btn_row_w.setStyleSheet("background: transparent;")
        btn_row_h = QHBoxLayout(btn_row_w); btn_row_h.setContentsMargins(4, 0, 4, 8); btn_row_h.setSpacing(6)
        _b = lambda text, cmd, color: (lambda b: (
            b.setFont(FONT_MAIN), b.setFixedSize(len(text)*8+20, 30),
            b.setStyleSheet(f"QPushButton {{ background: {BG}; border: 1px solid {color}; border-radius: 6px; color: {color}; }} QPushButton:hover {{ background: {color}; color: white; }}"),
            b.clicked.connect(cmd),
            btn_row_h.addWidget(b)
        ))(QPushButton(text))

        if code == 'tao':
            _b("Browse Tao data ⌕",      self._open_tao_browser,      SUCCESS)
        elif code == 'elegant':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse ELEGANT data ⌕",  self._open_elegant_browser,  SUCCESS)
        elif code == 'madx':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse MAD-X data ⌕",    self._open_madx_browser,     SUCCESS)
        elif code == 'xsuite':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse xsuite data ⌕",   self._open_xsuite_browser,   SUCCESS)
        btn_row_h.addStretch()
        inner_v.addWidget(btn_row_w)

        # Extra attrs — collapsed by default, toggle to show
        self.e_extra = QLineEdit(); self.e_extra.setFont(FONT_MONO)
        self.e_extra.setPlaceholderText("e.g.  k1, k2, angle")
        self.e_extra.setFixedWidth(380); self.e_extra.setStyleSheet(_ENTRY_SS)
        self.e_extra.setVisible(False)
        ex_btn = QPushButton("+ Extra attributes to fetch"); ex_btn.setFont(FONT_MAIN)
        ex_btn.setCheckable(True); ex_btn.setChecked(False)
        ex_btn.setStyleSheet(f"QPushButton {{ background: transparent; color: {ACCENT2}; border: none; text-align: left; padding: 2px 4px; }} QPushButton:checked {{ color: white; }} QPushButton:hover {{ color: white; }}")
        ex_btn.toggled.connect(lambda checked: self.e_extra.setVisible(checked))
        inner_v.addWidget(ex_btn)
        inner_v.addWidget(self.e_extra)

        # Y1
        lbl_y1 = QLabel("Y1 EXPRESSION  (left axis) — required"); lbl_y1.setFont(FONT_SEC)
        lbl_y1.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl_y1)
        _help(inner_v, "e.g.: k1 * beta_a  or  np.sqrt(beta_a * beta_b)")
        self.e_y1 = QLineEdit(); self.e_y1.setFont(FONT_MONO)
        self.e_y1.setPlaceholderText("e.g.  k1 * beta_a  or  s1, s2, s3  (comma-separated)")
        self.e_y1.setFixedWidth(380); self.e_y1.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_y1)
        r1_w = QWidget(); r1_w.setStyleSheet("background: transparent;")
        r1_h = QHBoxLayout(r1_w); r1_h.setContentsMargins(4, 0, 4, 8)
        lbl_y1l = QLabel("Y1 label:"); lbl_y1l.setFont(FONT_MAIN); lbl_y1l.setFixedWidth(80)
        lbl_y1l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
        self.e_y1_label = QLineEdit(); self.e_y1_label.setFont(FONT_MONO)
        self.e_y1_label.setPlaceholderText("axis label (defaults to expression)")
        self.e_y1_label.setFixedWidth(280); self.e_y1_label.setStyleSheet(_ENTRY_SS)
        r1_h.addWidget(lbl_y1l); r1_h.addWidget(self.e_y1_label); r1_h.addStretch()
        inner_v.addWidget(r1_w)

        # Y2
        lbl_y2 = QLabel("Y2 EXPRESSION  (right axis) — optional"); lbl_y2.setFont(FONT_SEC)
        lbl_y2.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl_y2)
        _help(inner_v, "e.g.: eta_x / np.sqrt(beta_a)  — leave blank for none.")
        self.e_y2 = QLineEdit(); self.e_y2.setFont(FONT_MONO)
        self.e_y2.setPlaceholderText("e.g.  eta_x / np.sqrt(beta_a)  (leave blank for none)")
        self.e_y2.setFixedWidth(380); self.e_y2.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_y2)
        r2_w = QWidget(); r2_w.setStyleSheet("background: transparent;")
        r2_h = QHBoxLayout(r2_w); r2_h.setContentsMargins(4, 0, 4, 8)
        lbl_y2l = QLabel("Y2 label:"); lbl_y2l.setFont(FONT_MAIN); lbl_y2l.setFixedWidth(80)
        lbl_y2l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
        self.e_y2_label = QLineEdit(); self.e_y2_label.setFont(FONT_MONO)
        self.e_y2_label.setPlaceholderText("axis label (defaults to expression)")
        self.e_y2_label.setFixedWidth(280); self.e_y2_label.setStyleSheet(_ENTRY_SS)
        r2_h.addWidget(lbl_y2l); r2_h.addWidget(self.e_y2_label); r2_h.addStretch()
        inner_v.addWidget(r2_w)
        inner_v.addStretch()

        # Focus tracking for insert-at-cursor — use focusChanged signal (safe on C++ objects)
        _tracked = [x for x in (self.e_extra, self.e_y1, self.e_y2) if x is not None]
        def _on_focus_changed(old_w, new_w):
            if new_w in _tracked:
                self._on_entry_focus(new_w)
        QApplication.instance().focusChanged.connect(_on_focus_changed)
        self._focus_conn = _on_focus_changed   # keep reference alive
        self._last_entry = self.e_y1

    def _on_entry_focus(self, entry):
        self._last_entry = entry
        if self._tao_browser and self._tao_browser.is_open():
            self._tao_browser.set_target(entry)

    def _open_tao_browser(self):
        if self._tao_browser and self._tao_browser.is_open():
            self._tao_browser.set_target(self._last_entry)
            self._tao_browser.lift(); return
        parent_win = self._inner_w.window()
        self._tao_browser = TaoDataBrowser(parent_win, target_entry=self._last_entry)

    def _open_elegant_browser(self):
        if self._elegant_browser and self._elegant_browser.is_open():
            self._elegant_browser.set_target(self._last_entry)
            self._elegant_browser.lift(); return
        parent_win = self._inner_w.window()
        # Resolve to absolute path so _parse_ele_outputs finds files relative to .ele location
        ele_abs = str(Path(self._input_file).resolve()) if self._input_file else ''
        self._elegant_browser = ElegantDataBrowser(
            parent_win, ele_file=ele_abs, target_entry=self._last_entry)

    def _open_madx_browser(self):
        if self._madx_browser and self._madx_browser.is_open():
            self._madx_browser.set_target(self._last_entry)
            self._madx_browser.lift(); return
        parent_win = self._inner_w.window()
        twiss_abs   = str(Path(self._input_file).resolve()) if self._input_file else ''
        survey_abs  = str(Path(self._madx_survey).resolve()) if self._madx_survey else None
        self._madx_browser = MadxDataBrowser(
            parent_win, twiss_file=twiss_abs, survey_file=survey_abs,
            target_entry=self._last_entry)

    def _open_xsuite_browser(self):
        if self._xsuite_browser and self._xsuite_browser.is_open():
            self._xsuite_browser.set_target(self._last_entry)
            self._xsuite_browser.lift(); return
        parent_win = self._inner_w.window()
        file_abs = str(Path(self._input_file).resolve()) if self._input_file else ''
        self._xsuite_browser = XsuiteDataBrowser(
            parent_win, json_file=file_abs,
            xsuite_twiss=self._xsuite_twiss, xsuite_line=self._xsuite_line,
            target_entry=self._last_entry)

    def _show_data_popup(self):
        if not self._input_file:
            QMessageBox.warning(None, "No Input File",
                "Please select an input file in the Input tab first."); return
        loading = QLabel("Loading lattice data…"); loading.setFont(FONT_MAIN)
        loading.setStyleSheet(f"color: {WARN}; background: transparent; padding: 2px 4px;")
        self._inner_v.insertWidget(self._inner_v.count() - 1, loading)

        class _Bridge(QObject):
            done = Signal(object)

        bridge = _Bridge()
        bridge.done.connect(lambda result: (loading.deleteLater(), self._open_data_popup(result)))

        input_abs  = str(Path(self._input_file).resolve()) if self._input_file else ''
        survey_abs = str(Path(self._madx_survey).resolve()) if getattr(self, '_madx_survey', None) else None

        def _worker():
            try:
                result = _inspect_available_data(
                    input_abs, self._code,
                    xsuite_twiss=getattr(self, '_xsuite_twiss', '4d'),
                    xsuite_line=getattr(self, '_xsuite_line', None),
                    madx_survey=survey_abs)
            except Exception as e:
                result = {'standard': [], 'extra': [], 'scalars': [],
                          'error': str(e)}
            bridge.done.emit(result)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_data_popup(self, result):
        if result.get('error'):
            QMessageBox.critical(None, "Lattice Load Error",
                f"Could not load lattice:\n\n{result['error'][:500]}"); return

        win = QDialog()
        win.setWindowTitle("Available Data")
        win.resize(600, 700)
        win.setStyleSheet(f"background: {BG}; color: {FG};")
        dv = QVBoxLayout(win); dv.setContentsMargins(16, 12, 16, 8)

        hdr = QLabel("AVAILABLE DATA"); hdr.setFont(FONT_SEC)
        hdr.setStyleSheet(f"color: {ACCENT2};")
        dv.addWidget(hdr)
        sub = QLabel("Click any name to insert into the focused field.")
        sub.setFont(FONT_SMALL); sub.setStyleSheet(f"color: {FG_DIM};")
        dv.addWidget(sub)

        sa, inner_w, inner_v = _make_scroll_widget()
        dv.addWidget(sa)

        def _section(title):
            sl = QLabel(title); sl.setFont(FONT_SEC)
            sl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 8px 2px 8px;")
            inner_v.addWidget(sl)
            line = QFrame(); line.setFrameShape(QFrame.HLine)
            line.setStyleSheet(f"background: {BORDER};")
            inner_v.addWidget(line)

        def _insert(name):
            entry = self._last_entry
            if entry is None: return
            cur = entry.text().strip()
            entry.setText((cur + ", " + name) if cur else name)

        def _item(name, desc, val=None):
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(8)
            nb = QPushButton(name); nb.setFont(FONT_MONO); nb.setFixedHeight(24)
            nb.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            nb.clicked.connect(lambda _=False, n=name: _insert(n))
            rh.addWidget(nb)
            if val is not None:
                vl = QLabel(str(val)); vl.setFont(FONT_MONO)
                vl.setStyleSheet(f"color: {SUCCESS}; background: transparent;")
                rh.addWidget(vl)
            if desc:
                dl = QLabel(desc); dl.setFont(FONT_SMALL)
                dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
                rh.addWidget(dl)
            rh.addStretch(); inner_v.addWidget(rw)

        _section("Standard Arrays  (always available)")
        for name, desc in result.get('standard', []):
            _item(name, desc)

        if result.get('extra'):
            _section("Extra Attributes  (add to \"Extra attributes\" field)")
            for item in result['extra']:
                name = item[0]; desc = item[1] if len(item) > 1 else ''
                _item(name, desc)

        inner_v.addStretch()
        close_b = QPushButton("Close"); close_b.setFont(FONT_MAIN); close_b.setFixedWidth(80)
        close_b.setStyleSheet(_BTN_SS); close_b.clicked.connect(win.close)
        dv.addWidget(close_b, alignment=Qt.AlignCenter)
        win.exec()

    def _parse(self, text):
        return [a.strip() for a in text.replace(',', ' ').split() if a.strip()]

    def _ok(self):
        y1_expr = self.e_y1.text().strip()
        if not y1_expr:
            QMessageBox.warning(None, "Y1 Required", "Please enter a Y1 expression."); return
        y2_expr = self.e_y2.text().strip()
        extra   = self._parse(self.e_extra.text()) if self.e_extra else []
        y1_label = self.e_y1_label.text().strip() or None
        y2_label = self.e_y2_label.text().strip() or None
        name     = self.e_name.text().strip() or "Expression Panel"
        self._on_done({
            'name': name, 'type': 'expr',
            'extra_attrs': extra,
            'y1_expr': y1_expr, 'y2_expr': y2_expr,
            'y1_label': y1_label, 'y2_label': y2_label,
        })

# ═══════════════════════════════════════════════════════════════════════════════
#  Data browsers — TaoDataBrowser, ElegantDataBrowser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_browser_window(title, width=440, height=720, parent=None):
    """Create a styled floating QDialog-style QWidget browser window."""
    win = QWidget(None, Qt.Window)
    win.setWindowTitle(title)
    win.resize(width, height)
    win.setStyleSheet(f"background: {BG}; color: {FG};")
    if parent:
        pg = parent.geometry()
        win.move(pg.right() + 10, pg.top())
    return win

def _make_browser_layout(win, title_text, subtitle_text=""):
    """Build header + search bar + scroll area for browser windows.
    Returns (main_vbox, inner_vbox, search_entry, all_items_list)."""
    vbox = QVBoxLayout(win); vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)

    # Header
    hdr = QWidget(); hdr.setFixedHeight(44)
    hdr.setStyleSheet(f"background: {PANEL};")
    hh = QHBoxLayout(hdr); hh.setContentsMargins(12, 0, 12, 0)
    tl = QLabel(title_text); tl.setFont(FONT_SEC); tl.setStyleSheet(f"color: {ACCENT2};")
    hh.addWidget(tl)
    if subtitle_text:
        sl = QLabel(subtitle_text); sl.setFont(FONT_SMALL); sl.setStyleSheet(f"color: {FG_DIM};")
        hh.addWidget(sl)
    hh.addStretch()
    vbox.addWidget(hdr)

    # Search bar
    sf = QWidget(); sf.setStyleSheet(f"background: {BG};"); sf.setFixedHeight(40)
    sh = QHBoxLayout(sf); sh.setContentsMargins(8, 4, 8, 4); sh.setSpacing(4)
    fl = QLabel("Filter:"); fl.setFont(FONT_MAIN); fl.setStyleSheet(f"color: {FG_LBL};")
    sh.addWidget(fl)
    search_e = QLineEdit(); search_e.setFont(FONT_MONO); search_e.setFixedWidth(200)
    search_e.setStyleSheet(_ENTRY_SS)
    clr_b = QPushButton("✕"); clr_b.setFont(FONT_MAIN); clr_b.setFixedSize(28, 24)
    clr_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {FG_DIM}; border: none; }} QPushButton:hover {{ color: {FG}; }}")
    clr_b.clicked.connect(lambda: search_e.clear())
    sh.addWidget(search_e); sh.addWidget(clr_b); sh.addStretch()
    vbox.addWidget(sf)

    # Scroll area
    sa, inner_w, inner_v = _make_scroll_widget()
    vbox.addWidget(sa, 1)

    all_items = []   # list of (row_widget, name, desc)

    def _filter_fn():
        q = search_e.text().lower()
        for row_w, name, desc in all_items:
            row_w.setVisible(not q or q in name.lower() or q in desc.lower())

    search_e.textChanged.connect(lambda _: _filter_fn())
    return vbox, inner_v, search_e, all_items

class TaoDataBrowser:
    """Floating window showing all Tao lat data types from Table 6.2."""

    def __init__(self, parent_window, target_entry=None):
        self._target = target_entry
        self._win    = None
        self._all_items = []
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "Tao Data Types — Table 6.2", 440, 720, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "Tao Data Types", "(lat source only)  •  click to insert")
        self._inner_v = inner_v
        self._build_content()
        # Footer
        foot = QLabel("Tao native names — auto-fetched when used in expressions")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _build_content(self):
        for cat, items in _TAO_DATA_CATEGORIES.items():
            # Category header
            cl = QLabel(cat); cl.setFont(FONT_SEC)
            cl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
            self._inner_v.addWidget(cl)
            line = QFrame(); line.setFrameShape(QFrame.HLine)
            line.setStyleSheet(f"background: {BORDER};")
            self._inner_v.addWidget(line)
            for name, desc in items:
                rw = self._make_item(name, desc)
                self._inner_v.addWidget(rw)
                self._all_items.append((rw, name, desc))
        self._inner_v.addStretch()

    def _make_item(self, name, desc):
        rw = QWidget(); rw.setStyleSheet("background: transparent;")
        rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
        b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
        b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
        b.clicked.connect(lambda _=False, n=name: self._insert(n))
        dl = QLabel(desc); dl.setFont(FONT_SMALL); dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
        return rw

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

def _parse_ele_outputs(ele_file):
    """Parse an ELEGANT .ele file and return output file paths.

    Returns dict with keys 'twi', 'cen', 'sig' — values are absolute
    paths or None if not configured in the .ele file.
    """
    result = {'twi': None, 'cen': None, 'sig': None}
    try:
        ele_dir = str(Path(ele_file).parent)
        with open(ele_file, 'r') as f:
            content = f.read()
        # Strip comments
        lines = []
        for line in content.splitlines():
            if '!' in line: line = line[:line.index('!')]
            lines.append(line)
        text = ' '.join(lines)

        # Helper: find filename= value in a namelist block
        def _find_param(block_name, param):
            pat = rf'&{block_name}.*?{param}\s*=\s*"([^"]+)"'
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m: return m.group(1)
            pat2 = rf'&{block_name}.*?{param}\s*=\s*(\S+)'
            m2 = re.search(pat2, text, re.IGNORECASE | re.DOTALL)
            if m2: return m2.group(1).rstrip(',')
            return None

        # run_setup provides rootname — used to expand %s
        # If not explicitly set, ELEGANT defaults to the .ele file stem
        rootname = _find_param('run_setup', 'rootname') or Path(ele_file).stem
        def _expand(p):
            if p is None: return None
            p = p.replace('%s', rootname)
            if not os.path.isabs(p):
                p = os.path.join(ele_dir, p)
            # ELEGANT appends .twi/.cen/.sig automatically if no extension
            return p

        # twiss_output filename
        twi = _find_param('twiss_output', 'filename')
        if twi:
            p = _expand(twi)
            result['twi'] = p if '.' in Path(p).name else p + '.twi'

        # centroid from run_setup
        cen = _find_param('run_setup', 'centroid')
        if cen:
            p = _expand(cen)
            result['cen'] = p if '.' in Path(p).name else p + '.cen'

        # sigma from run_setup
        sig = _find_param('run_setup', 'sigma')
        if sig:
            p = _expand(sig)
            result['sig'] = p if '.' in Path(p).name else p + '.sig'

    except Exception as _pe:
        result['_parse_error'] = str(_pe)
    return result


def _sddsquery(filepath):
    """Query an SDDS file using sddsquery.
    Returns (columns, parameters) where each is a list of (name, description) tuples.
    Returns ([], []) if sddsquery is not available or the file can't be read.

    sddsquery output format (fixed columns):
        NAME  UNITS  SYMBOL  FORMAT  TYPE  FIELD_LENGTH  DESCRIPTION
    The DESCRIPTION is the last whitespace-separated token(s) after FIELD_LENGTH.
    Columns and parameters are in separate sections separated by a blank line.
    """
    def _parse_section(lines):
        """Parse one section (columns or parameters) of sddsquery output.
        Skips the header line (NAME UNITS SYMBOL ...) and blank lines.
        Returns list of (name, description) tuples.
        """
        items = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip header line
            if stripped.startswith('NAME') and 'DESCRIPTION' in stripped:
                continue
            # Skip the units/format sub-header (second header line with just dashes or LENGTH)
            if stripped == 'LENGTH' or stripped.startswith('---'):
                continue
            # Split on whitespace — name is first token, description is last token(s)
            # Format: name  units  symbol  format  type  field_len  description
            # Description may be NULL or a multi-word string
            parts = stripped.split()
            if len(parts) < 1:
                continue
            name = parts[0]
            # Description: everything after the 6th field (field_length), or last token
            # If fewer than 7 fields, no description
            if len(parts) >= 7:
                desc = ' '.join(parts[6:])
                if desc == 'NULL':
                    desc = ''
            else:
                desc = ''
            items.append((name, desc))
        return items

    columns = []; parameters = []
    try:
        r = subprocess.run(['sddsquery', filepath],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError("sddsquery failed")

        # Split output into columns section and parameters section
        text = r.stdout
        col_section = []; par_section = []
        in_cols = False; in_params = False

        for line in text.splitlines():
            low = line.lower()
            if 'columns of data' in low or 'column of data' in low:
                in_cols = True; in_params = False; continue
            if 'parameters' in low and ('parameter' in low):
                in_params = True; in_cols = False; continue
            if in_cols:
                col_section.append(line)
            elif in_params:
                par_section.append(line)

        columns   = _parse_section(col_section)
        parameters = _parse_section(par_section)

    except Exception:
        # Fall back to -columnList / -parameterList for just names
        try:
            r = subprocess.run(['sddsquery', '-columnList', filepath],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                columns = [(l.strip(), '') for l in r.stdout.splitlines() if l.strip()]
        except Exception:
            pass
        try:
            r = subprocess.run(['sddsquery', '-parameterList', filepath],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                parameters = [(l.strip(), '') for l in r.stdout.splitlines() if l.strip()]
        except Exception:
            pass

    return columns, parameters


class ElegantDataBrowser:
    """Floating window showing available ELEGANT output data."""

    def __init__(self, parent_window, ele_file, target_entry=None):
        self._target   = target_entry
        self._win      = None
        self._all_items = []
        self._ele_file = ele_file
        self._outputs  = _parse_ele_outputs(ele_file) if ele_file else {}
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "ELEGANT Data Browser", 460, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "ELEGANT Data", "click to insert")
        self._inner_v = inner_v
        self._build_content()
        foot = QLabel("Columns are plottable vs s  •  Scalars are lattice-wide values")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _section(self, title, subtitle=""):
        lbl = QLabel(title + (f"  {subtitle}" if subtitle else ""))
        lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.addWidget(lbl)
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {BORDER};")
        self._inner_v.addWidget(line)

    def _unavailable_note(self, msg):
        lbl = QLabel(msg); lbl.setFont(FONT_SMALL)
        lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 16px 6px 16px;")
        self._inner_v.addWidget(lbl)

    def _add_items(self, items):
        for name, desc in items:
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=name: self._insert(n))
            dl = QLabel(desc); dl.setFont(FONT_SMALL); dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
            self._inner_v.addWidget(rw)
            self._all_items.append((rw, name, desc))

    def _build_content(self):
        # Show parse error or resolved .ele path for diagnostics
        if self._outputs.get('_parse_error'):
            self._unavailable_note(f"⚠  Parse error: {self._outputs['_parse_error']}")
        elif self._ele_file:
            self._unavailable_note(f"📄  {self._ele_file}")

        # Skip column — always present in SDDS, not a plottable quantity
        _skip_cols = {'ElementName', 'ElementType', 'ElementOccurence',
                      'ElementGroup', 'pCentral'}

        def _add_sdds_section(title, filepath, kind):
            """Add a section for one SDDS output file."""
            fname = Path(filepath).name if filepath else ''
            self._section(title, f"({fname})" if fname else "(not configured)")
            if not filepath:
                if kind == 'twi':
                    self._unavailable_note("⚠  No &twiss_output filename= found in .ele")
                elif kind == 'cen':
                    self._unavailable_note('⚠  Add centroid="%s.cen" to &run_setup to enable')
                elif kind == 'sig':
                    self._unavailable_note('⚠  Add sigma="%s.sig" to &run_setup to enable')
                return
            if not Path(filepath).exists():
                self._unavailable_note(f"⚠  File not found — run ELEGANT first\n    ({filepath})")
                return
            cols, params = _sddsquery(filepath)
            if not cols and not params:
                # sddsquery not available or failed — fall back to hardcoded list
                fallback = {
                    'twi': _ELEGANT_TWI_COLUMNS,
                    'cen': _ELEGANT_CEN_COLUMNS,
                    'sig': _ELEGANT_SIG_COLUMNS,
                }.get(kind, [])
                if fallback:
                    self._add_items(fallback)
                    self._unavailable_note("(sddsquery not found — showing default column list)")
                else:
                    self._unavailable_note("⚠  sddsquery not found — cannot read column names")
                return
            # Show columns
            col_items = [(name, desc) for name, desc in cols if name not in _skip_cols]
            if col_items:
                self._add_items(col_items)
            # Show parameters as a sub-section
            if params:
                self._section(f"  {title} — Parameters", "(lattice-wide scalars)")
                param_items = [(name, desc) for name, desc in params if name not in _skip_cols]
                if param_items:
                    self._add_items(param_items)

        _add_sdds_section("Twiss Columns",            self._outputs.get('twi'), 'twi')
        _add_sdds_section("Centroid Columns",         self._outputs.get('cen'), 'cen')
        _add_sdds_section("Sigma / Beam Size Columns", self._outputs.get('sig'), 'sig')

        self._inner_v.addStretch()

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


class MadxDataBrowser:
    """Floating browser showing available columns from MAD-X TFS files."""

    def __init__(self, parent_window, twiss_file='', survey_file=None,
                 target_entry=None):
        self._target      = target_entry
        self._win         = None
        self._all_items   = []
        self._twiss_file  = twiss_file
        self._survey_file = survey_file
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "MAD-X Data Browser", 480, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "MAD-X TFS Data", "click to insert")
        self._inner_v = inner_v
        self._build_content()
        foot = QLabel("Column names from your TFS files  •  click to insert into expression")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _section(self, title, subtitle=""):
        lbl = QLabel(title + (f"  {subtitle}" if subtitle else ""))
        lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.addWidget(lbl)
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {BORDER};")
        self._inner_v.addWidget(line)

    def _note(self, msg):
        lbl = QLabel(msg); lbl.setFont(FONT_SMALL)
        lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 16px 6px 16px;")
        self._inner_v.addWidget(lbl)

    def _add_items(self, items):
        for name, desc in items:
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=name: self._insert(n))
            dl = QLabel(desc); dl.setFont(FONT_SMALL)
            dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
            self._inner_v.addWidget(rw)
            self._all_items.append((rw, name, desc))

    def _build_content(self):
        _skip = {'NAME', 'KEYWORD', 'PARENT', 'TYPE', 'ORIGIN', 'COMMENTS'}

        # ── Twiss file ──────────────────────────────────────────────────────
        twi_name = Path(self._twiss_file).name if self._twiss_file else ''
        self._section("Twiss Columns", f"({twi_name})" if twi_name else "(no file)")
        if self._twiss_file and Path(self._twiss_file).exists():
            try:
                scalars, col_names, _ = _read_tfs(self._twiss_file)
                cols = [(c.lower(), f"twiss  ({c})") for c in col_names
                        if c.upper() not in _skip]
                if cols:
                    self._add_items(cols)
                else:
                    self._note("⚠  No data columns found in twiss file")
                # Scalars
                if scalars:
                    self._section("Twiss Scalars", "(header @ parameters)")
                    sc_items = [(k.lower(), f"{v}") for k, v in scalars.items()]
                    self._add_items(sc_items)
            except Exception as e:
                self._note(f"⚠  Could not read twiss file: {e}")
        elif self._twiss_file:
            self._note("⚠  Twiss file not found")
        else:
            self._note("⚠  No twiss file loaded")

        # ── Survey file ─────────────────────────────────────────────────────
        sv_name = Path(self._survey_file).name if self._survey_file else ''
        self._section("Survey Columns", f"({sv_name})" if sv_name else "(not loaded)")
        if self._survey_file and Path(self._survey_file).exists():
            try:
                _, sv_cols, _ = _read_tfs(self._survey_file)
                _skip_sv = {'NAME', 'KEYWORD', 'PARENT', 'TYPE'}
                sv_items = [(c.lower(), f"survey  ({c})") for c in sv_cols
                            if c.upper() not in _skip_sv]
                if sv_items:
                    self._add_items(sv_items)
                else:
                    self._note("⚠  No data columns found in survey file")
            except Exception as e:
                self._note(f"⚠  Could not read survey file: {e}")
        elif self._survey_file:
            self._note("⚠  Survey file not found")
        else:
            self._note("No survey file loaded  —  floor plan uses dead-reckoning")

        self._inner_v.addStretch()

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


class XsuiteDataBrowser:
    """Floating browser showing available columns from an xsuite twiss table."""

    def __init__(self, parent_window, json_file='', xsuite_twiss='4d',
                 xsuite_line=None, target_entry=None):
        self._target       = target_entry
        self._win          = None
        self._all_items    = []
        self._json_file    = json_file
        self._xsuite_twiss = xsuite_twiss
        self._xsuite_line  = xsuite_line
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "xsuite Data Browser", 480, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "xsuite Twiss Data", "click to insert")
        self._inner_v = inner_v

        # Loading label — replaced once data arrives
        self._loading_lbl = QLabel("Loading xsuite twiss table…")
        self._loading_lbl.setFont(FONT_MAIN)
        self._loading_lbl.setStyleSheet(f"color: {WARN}; background: transparent; padding: 8px;")
        self._inner_v.addWidget(self._loading_lbl)

        foot = QLabel("Column names from xsuite twiss  •  click to insert into expression")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

        # Load in background thread
        class _Bridge(QObject):
            done = Signal(object)
        self._bridge = _Bridge()
        self._bridge.done.connect(self._on_loaded)
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            data = load_xsuite(self._json_file, log_fn=None,
                               twiss_method=self._xsuite_twiss,
                               line_name=self._xsuite_line)
            _skip = {'s', 'elements', 'beam_params', '_tao', '_tw', 'name',
                     'element_type', 'isthick', 'parent_name'}
            cols = sorted([k for k in data.keys()
                           if k not in _skip
                           and isinstance(data.get(k), np.ndarray)])
            self._bridge.done.emit({'cols': cols, 'error': None})
        except Exception as e:
            self._bridge.done.emit({'cols': [], 'error': str(e)})

    def _on_loaded(self, result):
        if not (self._win and self._win.isVisible()):
            return
        self._loading_lbl.deleteLater()
        if result['error']:
            lbl = QLabel(f"⚠  Could not load xsuite file:\n{result['error'][:300]}")
            lbl.setFont(FONT_SMALL)
            lbl.setStyleSheet(f"color: {ERROR}; background: transparent; padding: 8px;")
            lbl.setWordWrap(True)
            self._inner_v.insertWidget(0, lbl)
            return
        cols = result['cols']
        if not cols:
            lbl = QLabel("⚠  No twiss columns found in file")
            lbl.setFont(FONT_SMALL)
            lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 8px;")
            self._inner_v.insertWidget(0, lbl)
            return
        # Add section header + items
        sec = QLabel("Twiss Columns"); sec.setFont(FONT_SEC)
        sec.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.insertWidget(0, sec)
        for i, col in enumerate(cols):
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(col); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=col: self._insert(n))
            rh.addWidget(b); rh.addStretch()
            self._inner_v.insertWidget(i + 1, rw)
            self._all_items.append((rw, col, ''))

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(FG))
    palette.setColor(QPalette.Base,            QColor(PANEL))
    palette.setColor(QPalette.AlternateBase,   QColor(BG))
    palette.setColor(QPalette.Text,            QColor(FG))
    palette.setColor(QPalette.Button,          QColor(PANEL))
    palette.setColor(QPalette.ButtonText,      QColor(FG))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(FG))
    palette.setColor(QPalette.PlaceholderText, QColor(FG_DIM))
    app.setPalette(palette)

    win = LuxV4GUI()
    win.show()
    sys.exit(app.exec())
