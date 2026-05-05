# =============================================================================
# core/gui.py — RanOptics main GUI (LuxV4GUI, _WorkerThread, _FodoLogo)
# =============================================================================

from __future__ import annotations
import fnmatch, json, math, os, re, threading, time, traceback
from pathlib import Path
import numpy as np

from PySide6.QtCore    import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui     import QAction, QColor, QFont, QPainter, QPen, QBrush, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTabWidget, QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

from core.themes import (
    ACCENT, ACCENT2, BG, BORDER, CRUST, ERROR, FG, FG_DIM, FG_LBL,
    MANTLE, PANEL, PEACH, RAN_CLR, SUCCESS, SURFACE2, HIGHLIGHT, WARN,
    FONT_BOLD, FONT_HDR, FONT_MAIN, FONT_MONO, FONT_SEC, FONT_SMALL,
    _BTN_SS, _CHK_SS, _COMBO_SS, _ENTRY_SS, _RB_SS, _SCROLL_SS, _TAB_SS,
)
from core.utils import (
    _clf, _make_scroll_widget,
    _sec, _card, _row, _lbl, _ent, _btn, _chk, _dd, _hint, _help, _rb,
    _parse_yrange, _parse_fp_range,
)
from core.engine import plot_optics
from core.loaders import load_tao, load_elegant, load_xsuite, _parse_tao_init
from core.overlays import (
    CustomPanelOverlay, ExprPanelOverlay,
    _TAO_DATA_CATEGORIES, _ELEGANT_TWI_COLUMNS, _ELEGANT_CEN_COLUMNS,
    _ELEGANT_SIG_COLUMNS, _ELEGANT_TWI_SCALARS,
)

# ── Worker thread ─────────────────────────────────────────────────────────────

class _WorkerThread(QThread):
    log_signal      = Signal(str, str)   # (text, tag)
    progress_signal = Signal(int, str)   # (pct, label)
    done_signal     = Signal(str)        # output_path
    error_signal    = Signal(str)        # traceback

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        self._fn()

# ── FODO logo widget ──────────────────────────────────────────────────────────

# ── Tao data browser categories ───────────────────────────────────────────────
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


class _FodoLogo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(108, 64)

    def paintEvent(self, event):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = 108, 64
        sx = W / 128.0; sy = H / 88.0

        def lx(x): return (x - 28) * sx
        def ly(y): return H - (y - 40) * sy

        from PySide6.QtCore import QPointF
        # betax curve
        pts_x = [QPointF(lx(32 + i / 80 * 104), ly(67 - 20 * math.cos(math.pi * 2 * i / 80)))
                 for i in range(81)]
        pen = QPen(QColor(RAN_CLR), 2); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for i in range(len(pts_x) - 1): p.drawLine(pts_x[i], pts_x[i + 1])
        # betay curve
        pts_y = [QPointF(lx(32 + i / 80 * 104), ly(67 + 20 * math.cos(math.pi * 2 * i / 80)))
                 for i in range(81)]
        pen2 = QPen(QColor(ERROR), 2); pen2.setDashPattern([4, 2]); pen2.setCapStyle(Qt.RoundCap)
        p.setPen(pen2)
        for i in range(len(pts_y) - 1): p.drawLine(pts_y[i], pts_y[i + 1])
        # Element bar
        ey = ly(106); eh = max(3, int(9 * sy))
        from PySide6.QtCore import QRectF
        for ex_, ew, col, lbl_text in [
            (30, 18, RAN_CLR, 'F'), (48, 30, None, None),
            (78, 18, ERROR, 'D'), (96, 30, None, None), (126, 18, RAN_CLR, 'F')
        ]:
            x1 = lx(ex_); x2 = lx(ex_ + ew)
            if col is None:
                p.setPen(QPen(QColor('#aaaaaa'), 1.2))
                my = ey - eh / 2
                p.drawLine(QPointF(x1, my), QPointF(x2, my))
            else:
                p.fillRect(int(x1), int(ey - eh), int(x2 - x1), int(eh), QColor(col))
                if lbl_text:
                    f = QFont("Monospace"); f.setPointSize(6); f.setBold(True)
                    p.setFont(f); p.setPen(QPen(QColor("white")))
                    p.drawText(QRectF(x1, ey - eh, x2 - x1, eh), Qt.AlignCenter, lbl_text)

# ── Main GUI class ────────────────────────────────────────────────────────────

class LuxV4GUI(QMainWindow):
    # Signals for cross-thread communication — emitting a Signal is always thread-safe
    _sig_log      = Signal(str, str)   # (text, tag)
    _sig_progress = Signal(int, str)   # (pct, label)
    _sig_done     = Signal(str)        # output_path
    _sig_failed   = Signal(str)        # traceback text
    _sig_finally  = Signal()           # always fires at end of run

    _PRESET_PANELS = [
        ('Twiss & Dispersion', 'twiss'), ('Beta Functions', 'beta'),
        ('Dispersion', 'dispersion'), ('Alpha Functions', 'alpha'),
        ('Orbit', 'orbit'), ('Phase Advance', 'phase'), ('Beam Size', 'beamsize'),
        ('Lattice Summary', 'summary'),
        ('Lattice Diff',    'latdiff'),
        ('Beamline Bar',    'bar'),
        ('Floor Plan X-Z',  'floor-xz'),
        ('Floor Plan Y-Z',  'floor-yz'),
    ]
    _RECENT_FILE = Path.home() / ".ranoptics_recent.json"
    _PRESET_FILE = Path.home() / ".ranoptics_presets.json"
    _MAX_RECENT  = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RanOptics — Accelerator Optics Plotter")
        self.resize(1280, 960); self.setMinimumSize(1000, 800)
        self.setStyleSheet(f"""
            QMainWindow {{ background: {BG}; }}
            QWidget {{ background: {BG}; }}
            QToolTip {{
                background: {PANEL}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding: 4px 8px;
            }}
        """)

        self._last_output = None
        self._worker      = None
        self._uni_checks  = {}   # {i: QCheckBox}
        self._uni_label_edits = {}  # {i: QLineEdit}
        self._uni_n       = 1
        self._panels      = [{'name': 'Floor Plan X-Z',      'spec': 'floor-xz'},
                              {'name': 'Twiss & Dispersion', 'spec': 'twiss'},
                              {'name': 'Beamline Bar',       'spec': 'bar'}]
        self._panel_rows  = []

        central = QWidget()
        self.setCentralWidget(central)
        self._root_layout = QVBoxLayout(central)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self._build_menubar()
        self._build_header()
        self._build_form()
        self._build_run_bar()
        self._build_log()
        self._build_statusbar()

        # Wire cross-thread signals to GUI slots
        self._sig_log.connect(self._log)
        self._sig_progress.connect(self._set_progress)
        self._sig_done.connect(self._on_run_done)
        self._sig_failed.connect(self._on_run_failed)
        self._sig_finally.connect(self._on_run_finally)

        self._refresh_recent_menu()
        self._refresh_preset_menu()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{
                background: {CRUST}; color: {FG_LBL};
                border-bottom: 1px solid {BORDER};
            }}
            QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
            QMenuBar::item:selected {{ background: {SURFACE2}; color: {FG}; }}
            QMenu {{
                background: {MANTLE}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {PANEL}; color: {ACCENT}; }}
            QMenu::separator {{ background: {BORDER}; height: 1px; margin: 4px 8px; }}
        """)

        # File menu
        fm = mb.addMenu("File")
        fm.addAction(QAction("Browse Input…",    self, triggered=self._browse_input))
        fm.addAction(QAction("Save Output As…",  self, triggered=self._browse_output))
        fm.addSeparator()
        self._recent_menu = fm.addMenu("Recent Files")
        fm.addSeparator()
        fm.addAction(QAction("Export CSV…",      self, triggered=self._export_csv))
        fm.addAction(QAction("Copy Output Path", self, triggered=self._copy_path))

        # Presets menu
        pm = mb.addMenu("Presets")
        pm.addAction(QAction("Save Current as Preset…", self, triggered=self._preset_save_dialog))
        pm.addSeparator()
        self._preset_menu = pm.addMenu("Load Preset")
        pm.addAction(QAction("Delete a preset…", self, triggered=self._preset_delete_dialog))

        # Run menu
        rm = mb.addMenu("Run")
        rm.addAction(QAction("▶ Run",       self, triggered=self._run))
        rm.addAction(QAction("🔍 Dry Run",  self, triggered=self._dry_run))
        rm.addAction(QAction("■ Cancel",    self, triggered=self._cancel))

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        h = QWidget(); h.setFixedHeight(64)
        h.setStyleSheet(f"background: {MANTLE}; border-bottom: 2px solid {BORDER};")
        row = QHBoxLayout(h); row.setContentsMargins(16, 0, 20, 0); row.setSpacing(8)

        row.addWidget(_FodoLogo())

        txt = QWidget(); txt.setStyleSheet(f"background: transparent;")
        tv  = QVBoxLayout(txt); tv.setContentsMargins(6, 8, 0, 8); tv.setSpacing(2)
        name_row = QWidget(); name_row.setStyleSheet("background: transparent;")
        nr = QHBoxLayout(name_row); nr.setContentsMargins(0,0,0,0); nr.setSpacing(4)
        ran = QLabel("Ran"); ran.setFont(FONT_HDR); ran.setStyleSheet(f"color: {RAN_CLR}; background: transparent; letter-spacing: 2px;")
        opt = QLabel("Optics"); opt.setFont(FONT_HDR); opt.setStyleSheet(f"color: {ERROR}; background: transparent; letter-spacing: 2px;")
        nr.addWidget(ran); nr.addWidget(opt); nr.addStretch()
        tv.addWidget(name_row)
        sub = QLabel("Accelerator Optics Plotter  •  v1.2.1"); sub.setFont(FONT_SMALL)
        sub.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        tv.addWidget(sub)
        row.addWidget(txt)
        row.addStretch()

        rf = QWidget(); rf.setStyleSheet("background: transparent;")
        rv = QVBoxLayout(rf); rv.setContentsMargins(0,0,0,0); rv.setSpacing(2)
        for t in ("Author: Randika Gamage  (randika@jlab.org)", "Support: ¯\\_(ツ)_/¯  (good luck, I believe in you)"):
            l = QLabel(t); l.setFont(FONT_SMALL); l.setAlignment(Qt.AlignLeft)
            l.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rv.addWidget(l)
        row.addWidget(rf)

        self._root_layout.addWidget(h)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = QWidget(); sb.setFixedHeight(28)
        sb.setStyleSheet(f"background: {MANTLE}; border-top: 1px solid {BORDER};")
        row = QHBoxLayout(sb); row.setContentsMargins(12, 0, 8, 0); row.setSpacing(8)
        self._status_lbl = QLabel("Idle"); self._status_lbl.setFont(FONT_SMALL)
        self._status_lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        row.addWidget(self._status_lbl); row.addStretch()
        self._progress = QProgressBar(); self._progress.setFixedWidth(180)
        self._progress.setFixedHeight(6); self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {CRUST}; border-radius: 3px; border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {HIGHLIGHT});
                border-radius: 3px;
            }}
        """)
        self._pct_lbl = QLabel(""); self._pct_lbl.setFont(FONT_BOLD)
        self._pct_lbl.setFixedWidth(40)
        self._pct_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        row.addWidget(self._progress); row.addWidget(self._pct_lbl)
        self._root_layout.addWidget(sb)

    def _set_status(self, text):
        self._status_lbl.setText(text)

    # ── Run bar ───────────────────────────────────────────────────────────────

    def _build_run_bar(self):
        bar = QWidget(); bar.setFixedHeight(52)
        bar.setStyleSheet(f"background: {MANTLE}; border-top: 1px solid {BORDER};")
        row = QHBoxLayout(bar); row.setContentsMargins(12, 6, 12, 6); row.setSpacing(6)

        self.run_btn = QPushButton("▶  Run"); self.run_btn.setFont(FONT_BOLD)
        self.run_btn.setFixedSize(100, 36); self.run_btn.clicked.connect(self._run)
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; border-radius: 8px;
                color: {CRUST}; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {HIGHLIGHT}; color: {CRUST}; }}
            QPushButton:disabled {{ background: {BORDER}; color: {FG_DIM}; border: none; }}
        """)
        row.addWidget(self.run_btn)

        def _action_btn(text, cmd, color, width=100):
            b = QPushButton(text); b.setFont(FONT_BOLD)
            b.setFixedSize(width, 36); b.clicked.connect(cmd)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {PANEL}; border: 1px solid {color};
                    border-radius: 8px; color: {color}; font-weight: 500;
                }}
                QPushButton:hover {{ background: {color}; color: {CRUST}; }}
                QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
            """)
            row.addWidget(b); return b

        self.stop_btn    = _action_btn("■  Cancel",    self._cancel,   ERROR)
        self.open_btn    = _action_btn("🌐  Open Plot", self._open_plot, SUCCESS, 130)
        self.dryrun_btn  = _action_btn("🔍  Dry Run",   self._dry_run,  ACCENT2, 115)
        self.stop_btn.setEnabled(False); self.open_btn.setEnabled(False)

        self.csv_btn = QPushButton("💾  Export CSV"); self.csv_btn.setFont(FONT_MAIN)
        self.csv_btn.setFixedSize(130, 36); self.csv_btn.clicked.connect(self._export_csv)
        self.csv_btn.setStyleSheet(f"""
            QPushButton {{
                background: {MANTLE}; border: 1px solid {BORDER};
                border-radius: 8px; color: {FG_LBL};
            }}
            QPushButton:hover {{ background: {SURFACE2}; color: {FG}; border-color: {ACCENT2}; }}
        """)
        row.addWidget(self.csv_btn)
        row.addStretch()

        clr = QPushButton("⊗  Clear log"); clr.setFont(FONT_MAIN)
        clr.setFixedSize(115, 36); clr.clicked.connect(self._clear_log)
        clr.setStyleSheet(self.csv_btn.styleSheet())
        row.addWidget(clr)

        self._root_layout.addWidget(bar)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log(self):
        self._log_autoscroll  = True
        self._log_filter      = 'all'   # 'all' | 'warn' | 'error'
        self._log_last_line   = ''      # for deduplication
        self._log_repeat_count = 0
        self._log_full        = []      # list of (text, tag) — full unfiltered history

        lf = QWidget(); lf.setStyleSheet(f"background: {BG};")
        lv = QVBoxLayout(lf); lv.setContentsMargins(12, 4, 12, 4); lv.setSpacing(2)

        # ── Log toolbar ───────────────────────────────────────────────────────
        tb = QWidget(); tb.setStyleSheet("background: transparent;")
        tbh = QHBoxLayout(tb); tbh.setContentsMargins(0, 0, 0, 2); tbh.setSpacing(6)

        hdr = QLabel("OUTPUT LOG"); hdr.setFont(FONT_SEC)
        hdr.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        tbh.addWidget(hdr)
        tbh.addStretch()

        # Filter dropdown
        self._log_filter_dd = QComboBox(); self._log_filter_dd.setFont(FONT_SMALL)
        self._log_filter_dd.addItems(["All", "Warnings+", "Errors only"])
        self._log_filter_dd.setFixedWidth(110); self._log_filter_dd.setStyleSheet(_COMBO_SS)
        self._log_filter_dd.currentIndexChanged.connect(self._on_log_filter_changed)
        tbh.addWidget(self._log_filter_dd)

        # Auto-scroll toggle
        self._log_scroll_btn = QPushButton("⇩ Auto"); self._log_scroll_btn.setFont(FONT_SMALL)
        self._log_scroll_btn.setFixedWidth(70); self._log_scroll_btn.setCheckable(True)
        self._log_scroll_btn.setChecked(True)
        self._log_scroll_btn.clicked.connect(self._on_log_scroll_toggle)
        self._log_scroll_btn.setStyleSheet(f"""
            QPushButton {{ background: {PANEL}; border: 1px solid {BORDER};
                border-radius: 6px; color: {FG_DIM}; padding: 2px 6px; }}
            QPushButton:checked {{ color: {ACCENT}; border-color: {ACCENT}; }}
            QPushButton:hover {{ background: {SURFACE2}; }}
        """)
        tbh.addWidget(self._log_scroll_btn)

        # Copy button
        copy_btn = QPushButton("⎘ Copy"); copy_btn.setFont(FONT_SMALL)
        copy_btn.setFixedWidth(70)
        copy_btn.clicked.connect(self._copy_log)
        copy_btn.setStyleSheet(f"""
            QPushButton {{ background: {PANEL}; border: 1px solid {BORDER};
                border-radius: 6px; color: {FG_DIM}; padding: 2px 6px; }}
            QPushButton:hover {{ background: {SURFACE2}; color: {FG}; }}
        """)
        tbh.addWidget(copy_btn)

        lv.addWidget(tb)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(FONT_MONO); self.log.setFixedHeight(140)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {MANTLE}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 6px; selection-background-color: {ACCENT};
            }}
        """)
        lv.addWidget(self.log)
        self._root_layout.addWidget(lf)
        self._log("Ready. Configure options above and click ▶ Run.\n", "dim")

    def _on_log_filter_changed(self, idx):
        self._log_filter = ['all', 'warn', 'error'][idx]
        self._redraw_log()

    def _on_log_scroll_toggle(self, checked):
        self._log_autoscroll = checked
        if checked:
            self.log.ensureCursorVisible()

    def _copy_log(self):
        text = self.log.toPlainText()
        QApplication.clipboard().setText(text)

    def _redraw_log(self):
        """Redraw log from history applying current filter."""
        self.log.clear()
        for text, tag in self._log_full:
            if self._log_filter == 'error' and tag not in ('error',):
                continue
            if self._log_filter == 'warn' and tag not in ('error', 'warn'):
                continue
            self._write_log(text, tag)

    # ── Form (tabs) ───────────────────────────────────────────────────────────

    def _build_form(self):
        outer = QWidget(); outer.setStyleSheet(f"background: {BG};")
        outer_h = QHBoxLayout(outer); outer_h.setContentsMargins(8, 4, 8, 0); outer_h.setSpacing(8)

        self._tab_l = QTabWidget(); self._tab_l.setStyleSheet(_TAB_SS); self._tab_l.setFont(FONT_SEC)
        self._tab_r = QTabWidget(); self._tab_r.setStyleSheet(_TAB_SS); self._tab_r.setFont(FONT_SEC)

        for name in ("Input", "Beam Settings"):   self._tab_l.addTab(QWidget(), name)
        for name in ("Panels", "Visual", "Export"): self._tab_r.addTab(QWidget(), name)

        # Wrap each tab's QWidget in a scroll area
        def _scroll_tab(tab_widget, idx):
            w = tab_widget.widget(idx)
            sa, inner, vbox = _make_scroll_widget()
            vbox.addStretch()
            layout = QVBoxLayout(w); layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(sa)
            return vbox

        self._input_layout  = _scroll_tab(self._tab_l, 0)
        self._beam_layout   = _scroll_tab(self._tab_l, 1)
        self._panels_layout = _scroll_tab(self._tab_r, 0)
        self._visual_layout = _scroll_tab(self._tab_r, 1)
        self._export_layout = _scroll_tab(self._tab_r, 2)

        outer_h.addWidget(self._tab_l, 1)
        outer_h.addWidget(self._tab_r, 1)
        self._root_layout.addWidget(outer, 1)

        # Remove trailing stretch from each, build content, re-add stretch
        for lay in (self._input_layout, self._beam_layout,
                    self._panels_layout, self._visual_layout, self._export_layout):
            lay.takeAt(lay.count() - 1)  # remove placeholder stretch

        self._build_input_section(self._input_layout)
        self._build_beam_section(self._beam_layout)
        self._build_panels_section(self._panels_layout)
        self._build_visual_section(self._visual_layout)
        self._build_export_section(self._export_layout)

        for lay in (self._input_layout, self._beam_layout,
                    self._panels_layout, self._visual_layout, self._export_layout):
            lay.addStretch(1)

    # ── Input section ─────────────────────────────────────────────────────────

    def _build_input_section(self, layout):
        r = _row(layout); _lbl(r, "Input file  *")
        self.w_input = _ent(r, width=220, placeholder="tao.init / run.ele / lattice.json")
        self.w_input.textChanged.connect(lambda t: self._on_input_change(t.strip()))
        _btn(r, "Browse", self._browse_input, width=70)
        _help(layout, "Auto-detected from extension: .init=Tao, .ele=ELEGANT, .json=xsuite.")

        r = _row(layout); _lbl(r, "Code backend")
        self.w_code = _dd(r, ["tao", "elegant", "xsuite", "madx"], width=110)
        self.w_code.currentTextChanged.connect(lambda _: (self._update_xsuite_rows(), self._update_madx_rows()))
        _help(layout, "Auto-detected from file extension. Override here if needed.")

        # xsuite extra rows (hidden initially)
        self._xsuite_widget = QWidget(); self._xsuite_widget.setStyleSheet("background: transparent;")
        xv = QVBoxLayout(self._xsuite_widget); xv.setContentsMargins(0, 0, 0, 0); xv.setSpacing(0)
        rx = _row(xv); _lbl(rx, "xsuite Twiss")
        self.w_xsuite_twiss = _dd(rx, ["4d", "6d"], width=80); _hint(rx, "4d = no RF")
        _help(xv, "4d: no RF (default). 6d: full longitudinal, requires RF cavities.")
        rl = _row(xv); _lbl(rl, "Line name")
        self.w_xsuite_line = _ent(rl, width=160, placeholder="e.g. ring  (auto-detect)")
        _help(xv, "Line name in xsuite JSON. Leave blank to auto-detect.")
        layout.addWidget(self._xsuite_widget)
        self._xsuite_widget.hide()

        # MAD-X extra row — survey file (hidden until madx selected)
        self._madx_widget = QWidget(); self._madx_widget.setStyleSheet("background: transparent;")
        mv = QVBoxLayout(self._madx_widget); mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
        rm = _row(mv); _lbl(rm, "Survey file (.tfs)")
        self.w_madx_survey = _ent(rm, width=200, placeholder="optional — for floor plan")
        _btn(rm, "Browse", self._browse_madx_survey, width=70)
        _help(mv, "MAD-X SURVEY output. Leave blank to use dead-reckoning for floor plan.")
        layout.addWidget(self._madx_widget)
        self._madx_widget.hide()

        # Universe selector (hidden initially)
        self._uni_widget = QWidget(); self._uni_widget.setStyleSheet("background: transparent;")
        self._uni_vbox = QVBoxLayout(self._uni_widget); self._uni_vbox.setContentsMargins(0, 0, 0, 0)
        self._uni_row_h = QHBoxLayout(); self._uni_row_h.setContentsMargins(8, 2, 8, 2)
        lbl_u = QLabel("Universes"); lbl_u.setFont(FONT_MAIN)
        lbl_u.setStyleSheet(f"color: {FG_LBL}; background: transparent;"); lbl_u.setFixedWidth(160)
        self._uni_row_h.addWidget(lbl_u)
        self._uni_checks_widget = QWidget(); self._uni_checks_widget.setStyleSheet("background: transparent;")
        self._uni_checks_h = QHBoxLayout(self._uni_checks_widget)
        self._uni_checks_h.setContentsMargins(0, 0, 0, 0); self._uni_checks_h.setSpacing(8)
        self._uni_row_h.addWidget(self._uni_checks_widget)
        w_uni_row = QWidget(); w_uni_row.setStyleSheet("background: transparent;")
        w_uni_row.setLayout(self._uni_row_h)
        self._uni_vbox.addWidget(w_uni_row)
        _help(self._uni_vbox, "Uncheck universes to exclude from the plot.")
        layout.addWidget(self._uni_widget)
        self._uni_widget.hide()

        r = _row(layout); _lbl(r, "Output HTML")
        self.w_output = _ent(r, width=180); self.w_output.setText("optics.html")
        _btn(r, "Save as", self._browse_output, width=70)
        _help(layout, "Output HTML file. Open in any browser.")

        _sec(layout, "Plot Settings")
        r = _row(layout); _lbl(r, "Layout mode")
        self.w_layout = _dd(r, ["panels", "floor"], width=110); _hint(r, "panels · floor")
        r = _row(layout); _lbl(r, "Range  START:END")
        self.w_range = _ent(r, width=220, placeholder="QUA01:QUA06  or  3.0:19.0")
        _help(layout, "Sub-range: element names (QUA01:QUA06) or s positions (3.0:19.0).")

        _sec(layout, "Tunnel Wall")
        r = _row(layout); _lbl(r, "Tunnel wall file")
        self.w_tunnel_file = _ent(r, width=200, placeholder="path/to/tunnel.dat")
        _btn(r, "Browse", self._browse_tunnel, width=70)
        _help(layout, "Overlay tunnel on floor plan. Format: x_in y_in z_in x_out y_out z_out.")

        _sec(layout, "Compare Files")
        _help(layout, "Load additional files to overlay or compare against the primary.")

        # List of compare file rows
        self._compare_files = []
        self._compare_list_w = QWidget(); self._compare_list_w.setStyleSheet("background: transparent;")
        self._compare_list_v = QVBoxLayout(self._compare_list_w)
        self._compare_list_v.setContentsMargins(8, 2, 8, 2); self._compare_list_v.setSpacing(3)
        layout.addWidget(self._compare_list_w)

        # Add button
        add_row = _row(layout)
        _btn(add_row, "+ Add file", self._add_compare_file, width=90, color=ACCENT2)

        # Mode + normalize
        r = _row(layout); _lbl(r, "Compare mode")
        self.w_compare_mode = _dd(r, ["Overlay", "Separate", "Difference", "Difference (%)"], width=140)
        r = _row(layout)
        self.w_normalize_s = _chk(r, "Normalize s (0→1)")
        _hint(r, "aligns lattices of different lengths")

    # ── Beam section ──────────────────────────────────────────────────────────

    def _build_beam_section(self, layout):
        _sec(layout, "Beam Size Parameters")

        r = _row(layout); _lbl(r, "Emittance type")
        self.w_emit_geo = QPushButton("Geometric"); self.w_emit_geo.setCheckable(True)
        self.w_emit_geo.setChecked(True)
        self.w_emit_norm = QPushButton("Normalized"); self.w_emit_norm.setCheckable(True)
        for b in (self.w_emit_geo, self.w_emit_norm):
            b.setFont(FONT_MAIN); b.setFixedSize(110, 30)
            b.setStyleSheet(f"""
                QPushButton {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG}; }}
                QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
                QPushButton:hover   {{ background: {BORDER}; }}
            """)
            r.addWidget(b)
        self.w_emit_geo.clicked.connect(lambda: (self.w_emit_norm.setChecked(False), self._update_emit_ui()))
        self.w_emit_norm.clicked.connect(lambda: (self.w_emit_geo.setChecked(False), self._update_emit_ui()))

        r = _row(layout); _lbl(r, "Emit-x  [m·rad]")
        self.w_emitx = _ent(r, width=140, placeholder="e.g.  1e-9")
        r = _row(layout); _lbl(r, "Emit-y  [m·rad]")
        self.w_emity = _ent(r, width=140, placeholder="e.g.  1e-9")
        r = _row(layout); _lbl(r, "σ_dp  (δp/p)")
        self.w_sigmadp = _ent(r, width=140, placeholder="e.g.  1e-3")
        r = _row(layout); _lbl(r, "n·σ  (beam size)")
        self.w_nsigma = _ent(r, width=60); self.w_nsigma.setText("1")

        # Normalized emittance extras (hidden until "Normalized" selected)
        self._norm_widget = QWidget(); self._norm_widget.setStyleSheet("background: transparent;")
        nv = QVBoxLayout(self._norm_widget); nv.setContentsMargins(0, 0, 0, 0); nv.setSpacing(0)

        r2 = _row(nv); _lbl(r2, "Particle")
        self.w_particle = _dd(r2, ["Electron", "Proton", "Muon", "Custom"], width=120)
        self.w_particle.currentTextChanged.connect(lambda _: self._update_emit_ui())
        _help(nv, "Auto-set for Electron/Proton/Muon. Choose Custom to enter manually.")

        r2 = _row(nv); _lbl(r2, "Beam energy [MeV]")
        self.w_energy = _ent(r2, width=140, placeholder="e.g.  100")
        self.w_energy.textChanged.connect(lambda _: self._update_betagamma())
        _help(nv, "Total energy in MeV. Used to compute βγ.")

        self._mass_widget = QWidget(); self._mass_widget.setStyleSheet("background: transparent;")
        mv = QVBoxLayout(self._mass_widget); mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
        rm2 = _row(mv); _lbl(rm2, "Rest mass [MeV/c²]")
        self.w_mass = _ent(rm2, width=140, placeholder="e.g.  938.3"); self.w_mass.setText("0.511")
        self.w_mass.textChanged.connect(lambda _: self._update_betagamma())
        nv.addWidget(self._mass_widget)

        r2 = _row(nv); _lbl(r2, "βγ  (computed)")
        self.w_betagamma = QLabel("—"); self.w_betagamma.setFont(FONT_MONO)
        self.w_betagamma.setStyleSheet(f"color: {SUCCESS}; background: transparent;")
        r2.addWidget(self.w_betagamma)

        layout.addWidget(self._norm_widget)
        self._norm_widget.hide()
        self._update_emit_ui()

    # ── Panels section ────────────────────────────────────────────────────────

    def _build_panels_section(self, layout):
        self._panels_layout_ref = layout   # save for overlay swap

        lbl = QLabel("Add panels, click name to rename:"); lbl.setFont(FONT_MAIN)
        lbl.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 12px 2px 12px;")
        layout.addWidget(lbl)
        _help(layout, "Stacked plots below the floor plan. ▲▼ to reorder, click name to rename.")

        self._panel_frame_widget = QWidget(); self._panel_frame_widget.setStyleSheet("background: transparent;")
        self._panel_frame_vbox = QVBoxLayout(self._panel_frame_widget)
        self._panel_frame_vbox.setContentsMargins(8, 4, 8, 4); self._panel_frame_vbox.setSpacing(2)
        layout.addWidget(self._panel_frame_widget)
        self._render_panel_list()

        lbl2 = QLabel("Add preset:"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 12px 2px 12px;")
        layout.addWidget(lbl2)

        _NCOLS = 3
        btn_grid_w = QWidget(); btn_grid_w.setStyleSheet("background: transparent;")
        btn_grid = QGridLayout(btn_grid_w); btn_grid.setContentsMargins(8, 0, 8, 4); btn_grid.setSpacing(4)
        for col in range(_NCOLS): btn_grid.setColumnStretch(col, 1)
        # Preset panels
        for i, (name, key) in enumerate(self._PRESET_PANELS):
            b = QPushButton(name); b.setFont(FONT_MAIN); b.setFixedHeight(30)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {MANTLE}; border: 1px solid {BORDER};
                    border-radius: 8px; color: {FG_LBL};
                }}
                QPushButton:hover {{ background: {SURFACE2}; color: {ACCENT}; border-color: {ACCENT}; }}
            """)
            b.clicked.connect(lambda _=False, n=name, k=key: self._add_preset_panel(n, k))
            btn_grid.addWidget(b, i // _NCOLS, i % _NCOLS)
        # Custom and Expression panel buttons in the same grid
        n_presets = len(self._PRESET_PANELS)
        for j, (text, cmd) in enumerate([
            ("+ Custom panel...",     self._add_custom_panel_dialog),
            ("+ Expression panel...", self._add_expr_panel_dialog),
        ]):
            idx = n_presets + j
            b = QPushButton(text); b.setFont(FONT_MAIN); b.setFixedHeight(30)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {MANTLE}; border: 1px solid {ACCENT2};
                    border-radius: 8px; color: {ACCENT2};
                }}
                QPushButton:hover {{ background: {ACCENT2}; color: {CRUST}; }}
            """)
            b.clicked.connect(cmd)
            btn_grid.addWidget(b, idx // _NCOLS, idx % _NCOLS)
        layout.addWidget(btn_grid_w)

        _sec(layout, "Panel Options")
        r = _row(layout)
        self.w_show_tune = _chk(r, "Show tune & chromaticity")
        _help(layout, "Annotates Qₓ, Qᵧ, Qₓ', Qᵧ' on the first panel.")
        r = _row(layout)
        self.w_show_titles = _chk(r, "Show panel titles"); self.w_show_titles.setChecked(True)
        r = _row(layout)
        self.w_bar_lite = _chk(r, "Beamline bar lite")
        _help(layout, "Faster beamline bar rendering for large lattices. Uses the same two-trace method as the floor plan.")
        r = _row(layout); _lbl(r, "Panel spacing (px)")
        self.w_panel_spacing = _ent(r, width=60, placeholder="80"); self.w_panel_spacing.setText("80")
        _hint(r, "pixels between panels")

    # ── Visual section ────────────────────────────────────────────────────────

    def _build_visual_section(self, layout):
        # ── General ──────────────────────────────────────────────────────────
        r = _row(layout); _lbl(r, "Plot title")
        self.w_title = _ent(r, width=200, placeholder="optional")
        r = _row(layout); _lbl(r, "Aspect ratio  W:H")
        self.w_aspect = _ent(r, width=80, placeholder="e.g.  1:2")

        # ── Floor Plan ───────────────────────────────────────────────────────
        _sec(layout, "Floor Plan")
        r = _row(layout); _lbl(r, "X-Z elem ratio")
        self.w_elem_h = _ent(r, width=60); self.w_elem_h.setText("0.05")
        _hint(r, "fraction of axis span")
        r = _row(layout); _lbl(r, "Y-Z elem ratio")
        self.w_elem_h_yz = _ent(r, width=60); _hint(r, "blank = same as X-Z")
        r = _row(layout); _lbl(r, "XZ Y-range")
        self.w_fp_xz_range = _ent(r, width=100, placeholder="-0.5:0.5"); _hint(r, "blank = auto")
        r = _row(layout); _lbl(r, "YZ Y-range")
        self.w_fp_yz_range = _ent(r, width=100, placeholder="-1:1"); _hint(r, "blank = auto")
        r = _row(layout)
        self.w_show_xz = _chk(r, "Show X-Z"); self.w_show_xz.setChecked(True)
        self.w_show_yz = _chk(r, "Show Y-Z"); self.w_show_yz.setChecked(True)
        _hint(r, "(floor mode only)")

        # ── Display ──────────────────────────────────────────────────────────
        _sec(layout, "Display")
        _disp_w = QWidget(); _disp_w.setStyleSheet("background: transparent;")
        _disp_g = QGridLayout(_disp_w)
        _disp_g.setContentsMargins(8, 2, 8, 4); _disp_g.setSpacing(4)
        _disp_g.setColumnStretch(0, 1); _disp_g.setColumnStretch(1, 1); _disp_g.setColumnStretch(2, 1)
        _chk_items = [
            ("No labels",          lambda c: setattr(self, 'w_no_labels',        c)),
            ("Flip bends",         lambda c: setattr(self, 'w_flip_bend',         c)),
            ("Dark mode",          lambda c: setattr(self, 'w_dark',              c)),
            ("Color beampipes",    lambda c: setattr(self, 'w_color_beampipes',   c)),
            ("Show tunnel",        lambda c: setattr(self, 'w_show_tunnel',       c)),
            ("Legend inside",      lambda c: setattr(self, 'w_legend_inside',     c)),
            ("Markers in floor",   lambda c: setattr(self, 'w_show_markers',      c)),
            ("Markers in bar",     lambda c: setattr(self, 'w_show_markers_bar',  c)),
        ]
        for i, (lbl_txt, setter) in enumerate(_chk_items):
            cb = QCheckBox(lbl_txt); cb.setFont(FONT_MAIN); cb.setStyleSheet(_CHK_SS)
            setter(cb)
            _disp_g.addWidget(cb, i // 3, i % 3)
        layout.addWidget(_disp_w)

        # ── Font Sizes ───────────────────────────────────────────────────────
        _sec(layout, "Font Sizes")
        _fs_w = QWidget(); _fs_w.setStyleSheet("background: transparent;")
        _fs_g = QGridLayout(_fs_w)
        _fs_g.setContentsMargins(8, 2, 8, 4); _fs_g.setSpacing(6)
        _fs_g.setColumnStretch(0, 1); _fs_g.setColumnStretch(1, 1); _fs_g.setColumnStretch(2, 1)
        for col, (lbl_txt, placeholder, attr) in enumerate([
                ("Axis labels",  "12", 'w_fs_axis'),
                ("Tick labels",  "10", 'w_fs_tick'),
                ("Panel titles", "13", 'w_fs_title'),
                ("Annotations",  "8",  'w_fs_annot'),
                ("Legend",       "10", 'w_fs_legend'),
        ]):
            r_idx = col // 3; c_idx = col % 3
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            row_h = QHBoxLayout(row_w); row_h.setContentsMargins(0, 0, 0, 0); row_h.setSpacing(4)
            l = QLabel(lbl_txt); l.setFont(FONT_SMALL)
            l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
            e = QLineEdit(); e.setFont(FONT_MONO); e.setFixedWidth(42); e.setFixedHeight(24)
            e.setPlaceholderText(placeholder); e.setStyleSheet(_ENTRY_SS)
            row_h.addWidget(l); row_h.addWidget(e); row_h.addStretch()
            setattr(self, attr, e)
            _fs_g.addWidget(row_w, r_idx, c_idx)
        layout.addWidget(_fs_w)
        _help(layout, "Leave blank to use Plotly defaults.")

    # ── Export section ────────────────────────────────────────────────────────

    def _build_export_section(self, layout):
        r = _row(layout)
        self.w_png = _chk(r, "Save PNG"); self.w_pdf = _chk(r, "Save PDF")
        _hint(r, "requires: pip install kaleido")
        _help(layout, "Requires: pip install kaleido")
        r = _row(layout); _lbl(r, "DPI (PNG)")
        self.w_dpi = _ent(r, width=60); self.w_dpi.setText("300")
        _help(layout, "PNG resolution. 300 DPI = publication quality.")

        _sec(layout, "CSV Export")
        r = _row(layout); _lbl(r, "CSV base name")
        self.w_csv_base = _ent(r, width=160, placeholder="lattice")
        self.w_csv_base.setText("lattice")
        _help(layout, "e.g. 'ltr' → ltr-twiss.csv, ltr-orbit.csv, ...")

    # ── Panel list rendering ──────────────────────────────────────────────────

    def _render_panel_list(self):
        # Save legend position state before clearing
        _saved_leg = {}
        if hasattr(self, '_panel_leg_checks'):
            for k, btn in self._panel_leg_checks.items():
                try:
                    if btn.isChecked():
                        xe = self._panel_leg_x_edits.get(k)
                        ye = self._panel_leg_y_edits.get(k)
                        _saved_leg[k] = (xe.text() if xe else '', ye.text() if ye else '')
                except RuntimeError:
                    pass
        for w in self._panel_rows:
            w.setParent(None); w.deleteLater()
        self._panel_rows = []
        if not hasattr(self, '_panel_height_edits'):
            self._panel_height_edits = {}
        # Ensure every dict-spec panel has a unique _id regardless of how it was added
        import uuid as _uuid
        for _p in self._panels:
            _s = _p.get('spec', '')
            if isinstance(_s, dict) and '_id' not in _s:
                _s['_id'] = _uuid.uuid4().hex[:8]
        self._panel_leg_checks  = {}
        self._panel_leg_x_edits = {}
        self._panel_leg_y_edits = {}
        self._panel_annot_checks = {}
        self._panel_annot_edits  = {}
        n = len(self._panels)
        _DEFAULT_H = {'floor-xz': 220, 'floor-yz': 220, 'bar': 80,
                      'latdiff': 260, 'summary': 260}
        for pos, panel in enumerate(self._panels):
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(row_w); rh.setContentsMargins(0, 1, 0, 1); rh.setSpacing(4)

            name_btn = QPushButton(panel['name']); name_btn.setFont(FONT_MAIN)
            name_btn.setFixedWidth(155); name_btn.setFixedHeight(26)
            name_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; color: {FG}; text-align: left; padding: 2px 6px; }}
                QPushButton:hover {{ background: {PANEL}; border-radius: 4px; }}
            """)
            name_btn.clicked.connect(lambda _=False, p=pos: self._rename_panel(p))
            rh.addWidget(name_btn)

            # Height field — keyed by _id for custom/expr panels, spec string for presets
            spec = panel.get('spec', '')
            if isinstance(spec, str):
                h_key    = spec
                spec_key = spec
            else:
                h_key    = spec.get('_id', spec.get('type', 'custom'))
                spec_key = spec.get('type', 'custom')
            default_h = str(_DEFAULT_H.get(spec_key, 280))
            prev_val = self._panel_height_edits.get(h_key, default_h)
            h_edit = QLineEdit(prev_val); h_edit.setFixedWidth(55); h_edit.setFixedHeight(28)
            h_edit.setFont(FONT_MAIN); h_edit.setStyleSheet(_ENTRY_SS)
            h_edit.setToolTip("Panel height in pixels")
            self._panel_height_edits[h_key] = prev_val
            h_edit.textChanged.connect(lambda v, k=h_key: self._panel_height_edits.update({k: v}))
            rh.addWidget(h_edit)

            for sym, cmd, col in [
                ("▲", lambda _=False, p=pos: self._move_panel(p, -1), ACCENT),
                ("▼", lambda _=False, p=pos: self._move_panel(p, +1), ACCENT),
                ("✕", lambda _=False, p=pos: self._remove_panel(p),   ERROR),
            ]:
                disabled = (sym == "▲" and pos == 0) or (sym == "▼" and pos == n - 1)
                b = QPushButton(sym); b.setFixedSize(26, 26); b.setFont(FONT_SMALL)
                b.setEnabled(not disabled)
                b.setStyleSheet(f"""
                    QPushButton {{ background: {BORDER}; border-radius: 4px; color: {FG}; border: none; }}
                    QPushButton:hover:enabled {{ background: {col}; color: white; }}
                    QPushButton:disabled {{ color: {FG_DIM}; }}
                """)
                b.clicked.connect(cmd)
                rh.addWidget(b)
            # Annotation toggle — not available for floor plan panels
            _no_annot_panels = {'floor-xz', 'floor-yz'}
            _spec_str = spec if isinstance(spec, str) else spec.get('type', '')
            if _spec_str not in _no_annot_panels:
                annot_on = bool(panel.get('annot_pattern', ''))
                annot_btn = QPushButton('✎'); annot_btn.setFixedSize(26, 26)
                annot_btn.setFont(FONT_SMALL); annot_btn.setCheckable(True)
                annot_btn.setChecked(annot_on)
                annot_btn.setToolTip('Annotate elements (wildcard pattern)')
                annot_btn.setStyleSheet(f"""
                    QPushButton {{ background: {BORDER}; border-radius: 4px; color: {FG}; border: none; }}
                    QPushButton:checked {{ background: {ACCENT2}; color: white; }}
                    QPushButton:hover {{ background: {ACCENT2}; color: white; }}
                """)
                rh.addWidget(annot_btn)

                # Pattern field — visible only when toggled on
                annot_edit = QLineEdit(panel.get('annot_pattern', ''))
                annot_edit.setFixedWidth(90); annot_edit.setFixedHeight(22)
                annot_edit.setFont(FONT_MAIN); annot_edit.setStyleSheet(_ENTRY_SS)
                annot_edit.setPlaceholderText('e.g. IPM*')
                annot_edit.setVisible(annot_on)
                rh.addWidget(annot_edit)

                def _on_annot_toggle(checked, p=pos, ae=annot_edit):
                    ae.setVisible(checked)
                    self._panels[p]['annot_pattern'] = ae.text().strip() if checked else ''
                def _on_annot_text(text, p=pos, ab=annot_btn):
                    if ab.isChecked():
                        self._panels[p]['annot_pattern'] = text.strip()
                annot_btn.toggled.connect(_on_annot_toggle)
                annot_edit.textChanged.connect(_on_annot_text)

                self._panel_annot_checks[pos] = annot_btn
                self._panel_annot_edits[pos]  = annot_edit

            # Legend position toggle — not for bar, summary, latdiff
            _no_leg_panels = {'bar', 'summary', 'latdiff'}
            if _spec_str not in _no_leg_panels:
                _PRESET_SPECS = {'twiss','beta','dispersion','alpha','orbit','phase','beamsize','twiss_disp','floor-xz','floor-yz','bar','summary','latdiff'}
                _leg_key = _spec_str if _spec_str in _PRESET_SPECS else panel['name']
                leg_btn = QPushButton('⊹'); leg_btn.setFixedSize(26, 26)
                leg_btn.setFont(FONT_SMALL); leg_btn.setCheckable(True)
                leg_btn.setChecked(False)
                leg_btn.setToolTip('Set legend position (x:y, normalized 0-1)')
                leg_btn.setStyleSheet(f"""
                    QPushButton {{ background: {BORDER}; border-radius: 4px; color: {FG}; border: none; }}
                    QPushButton:checked {{ background: {HIGHLIGHT}; color: {CRUST}; }}
                    QPushButton:hover {{ background: {HIGHLIGHT}; color: {CRUST}; }}
                """)
                rh.addWidget(leg_btn)

                leg_x_edit = QLineEdit(); leg_x_edit.setFixedWidth(36); leg_x_edit.setFixedHeight(22)
                leg_x_edit.setFont(FONT_MAIN); leg_x_edit.setStyleSheet(_ENTRY_SS)
                leg_x_edit.setPlaceholderText('X')
                leg_x_edit.setVisible(False)
                rh.addWidget(leg_x_edit)

                leg_y_edit = QLineEdit(); leg_y_edit.setFixedWidth(36); leg_y_edit.setFixedHeight(22)
                leg_y_edit.setFont(FONT_MAIN); leg_y_edit.setStyleSheet(_ENTRY_SS)
                leg_y_edit.setPlaceholderText('Y')
                leg_y_edit.setVisible(False)
                rh.addWidget(leg_y_edit)

                def _on_leg_toggle(checked, xe=leg_x_edit, ye=leg_y_edit):
                    xe.setVisible(checked)
                    ye.setVisible(checked)
                # Restore saved state if available
                if _leg_key in _saved_leg:
                    _sx, _sy = _saved_leg[_leg_key]
                    leg_btn.setChecked(True)
                    leg_x_edit.setText(_sx)
                    leg_y_edit.setText(_sy)
                    leg_x_edit.setVisible(True)
                    leg_y_edit.setVisible(True)

                leg_btn.toggled.connect(_on_leg_toggle)

                self._panel_leg_checks[_leg_key]  = leg_btn
                self._panel_leg_x_edits[_leg_key] = leg_x_edit
                self._panel_leg_y_edits[_leg_key] = leg_y_edit

            rh.addStretch()
            self._panel_frame_vbox.addWidget(row_w)
            self._panel_rows.append(row_w)

    def _move_panel(self, pos, d):
        new = pos + d
        if 0 <= new < len(self._panels):
            self._panels[pos], self._panels[new] = self._panels[new], self._panels[pos]
        self._render_panel_list()

    def _remove_panel(self, pos):
        if len(self._panels) > 1:
            self._panels.pop(pos)
        else:
            QMessageBox.warning(self, "Cannot Remove", "At least one panel must remain.")
        self._render_panel_list()

    def _rename_panel(self, pos):
        name, ok = QInputDialog.getText(self, "Rename Panel", "Panel name:",
                                         text=self._panels[pos]['name'])
        if ok and name.strip():
            self._panels[pos]['name'] = name.strip()
            self._render_panel_list()

    def _get_panels(self):
        return [p['spec'] for p in self._panels] if self._panels else ['twiss']

    def _get_panel_annotations(self):
        """Return {panel_index: pattern} for panels with annot_pattern set."""
        result = {}
        for i, p in enumerate(self._panels):
            pat = p.get('annot_pattern', '').strip()
            if pat:
                result[i] = pat
        return result or None

    def _get_legend_positions(self):
        """Return {pos_key: [x, y]} by reading live legend position fields directly."""
        if not hasattr(self, '_panel_leg_checks'):
            return None
        result = {}
        for key, btn in self._panel_leg_checks.items():
            if btn.isChecked():
                xe = self._panel_leg_x_edits.get(key)
                ye = self._panel_leg_y_edits.get(key)
                if xe and ye:
                    try:
                        x = float(xe.text().strip())
                        y = float(ye.text().strip())
                        result[key] = [x, y]
                    except (ValueError, TypeError):
                        pass
        return result if result else None

    def _get_panel_heights(self):
        if not hasattr(self, '_panel_height_edits') or not self._panel_height_edits:
            return None
        result = {}
        for spec, val in self._panel_height_edits.items():
            try:
                result[spec] = int(val)
            except (ValueError, TypeError):
                pass
        return result if result else None

    def _add_preset_panel(self, name, key):
        existing = [p['spec'] for p in self._panels if isinstance(p['spec'], str)]
        if key in existing:
            r = QMessageBox.question(self, "Duplicate Panel",
                                     f"'{name}' already in list. Add again?")
            if r != QMessageBox.Yes: return
        self._panels.append({'name': name, 'spec': key})
        self._render_panel_list()

    def _add_expr_panel_dialog(self):
        self._push_overlay(lambda container, done:
            ExprPanelOverlay(container, done,
                             code=self.w_code.currentText(),
                             input_file=self.w_input.text().strip(),
                             xsuite_twiss=self.w_xsuite_twiss.currentText(),
                             xsuite_line=self.w_xsuite_line.text().strip(),
                             madx_survey=self.w_madx_survey.text().strip() or None))

    def _add_custom_panel_dialog(self):
        self._push_overlay(lambda container, done:
            CustomPanelOverlay(container, done))

    def _push_overlay(self, builder_fn):
        """Replace panels tab content with an overlay widget."""
        tab_widget = self._tab_r.widget(0)   # "Panels" tab
        # Hide existing layout widget
        old_sa = tab_widget.layout().itemAt(0).widget()
        old_sa.hide()

        overlay_w = QWidget(); overlay_w.setStyleSheet(f"background: {BG};")
        overlay_v = QVBoxLayout(overlay_w); overlay_v.setContentsMargins(0, 0, 0, 0)
        tab_widget.layout().addWidget(overlay_w)

        def _on_done(result):
            overlay_w.hide(); overlay_w.setParent(None); overlay_w.deleteLater()
            old_sa.show()
            if result:
                import uuid as _uuid
                if isinstance(result, dict) and '_id' not in result:
                    result['_id'] = _uuid.uuid4().hex[:8]
                self._panels.append({'name': result['name'], 'spec': result})
                self._render_panel_list()

        builder_fn(overlay_v, _on_done)

    # ── Compare file management ───────────────────────────────────────────────

    def _add_compare_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select compare file", "",
            "All supported (*.init *.ele *.json);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog)
        if not f: return
        ext = Path(f).suffix.lower()
        code = {'.init': 'tao', '.ele': 'elegant', '.json': 'xsuite'}.get(ext, 'tao')
        entry = {'file': f, 'code': code, 'label': Path(f).stem,
                 'uni_n': 1, 'uni_labels': {}, 'uni_checks': {}}
        if ext == '.init':
            try:
                n, labels = _parse_tao_init(f)
                entry['uni_n'] = n
                entry['uni_labels'] = labels
            except Exception:
                pass
        self._compare_files.append(entry)
        self._render_compare_list()

    def _render_compare_list(self):
        # Clear existing rows
        while self._compare_list_v.count():
            item = self._compare_list_v.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for i, entry in enumerate(self._compare_files):
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            row_h = QHBoxLayout(row_w); row_h.setContentsMargins(0, 0, 0, 0); row_h.setSpacing(4)

            # Label (editable)
            lbl_e = QLineEdit(entry['label']); lbl_e.setFont(FONT_MONO)
            lbl_e.setFixedWidth(100); lbl_e.setStyleSheet(_ENTRY_SS)
            lbl_e.setToolTip("Display label")
            lbl_e.textChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'label': t}))
            row_h.addWidget(lbl_e)

            # File path (truncated display)
            path_lbl = QLabel(Path(entry['file']).name); path_lbl.setFont(FONT_SMALL)
            path_lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            path_lbl.setToolTip(entry['file'])
            row_h.addWidget(path_lbl)

            # Code badge
            code_dd = QComboBox(); code_dd.setFont(FONT_SMALL)
            code_dd.addItems(["tao", "elegant", "xsuite"])
            code_dd.setCurrentText(entry['code']); code_dd.setFixedWidth(82)
            code_dd.setStyleSheet(_COMBO_SS)
            code_dd.currentTextChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'code': t}))
            row_h.addWidget(code_dd)

            # Remove button
            rm = QPushButton("✕"); rm.setFixedSize(22, 22); rm.setFont(FONT_SMALL)
            rm.setStyleSheet(f"QPushButton {{ background: {BORDER}; color: {FG_DIM}; border: none; border-radius: 3px; }}"
                             f"QPushButton:hover {{ background: {ERROR}; color: white; }}")
            rm.clicked.connect(lambda _=False, idx=i: self._remove_compare_file(idx))
            row_h.addWidget(rm)
            row_h.addStretch()
            self._compare_list_v.addWidget(row_w)

            # Universe checkboxes for multi-universe Tao files
            n = entry.get('uni_n', 1)
            if n > 1:
                uni_row_w = QWidget(); uni_row_w.setStyleSheet("background: transparent;")
                uni_row_h = QHBoxLayout(uni_row_w)
                uni_row_h.setContentsMargins(12, 0, 0, 2); uni_row_h.setSpacing(6)
                ul = QLabel("Universes:"); ul.setFont(FONT_SMALL)
                ul.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
                uni_row_h.addWidget(ul)
                uni_checks = {}
                labels = entry.get('uni_labels', {})
                for ui in range(1, n + 1):
                    cb = QCheckBox(f"u{ui}:{labels.get(ui, f'u{ui}')}")
                    cb.setChecked(True); cb.setFont(FONT_SMALL); cb.setStyleSheet(_CHK_SS)
                    uni_checks[ui] = cb
                    uni_row_h.addWidget(cb)
                uni_row_h.addStretch()
                self._compare_list_v.addWidget(uni_row_w)
                self._compare_files[i]['uni_checks'] = uni_checks

    def _remove_compare_file(self, idx):
        if 0 <= idx < len(self._compare_files):
            self._compare_files.pop(idx)
            self._render_compare_list()

    def _get_compare_list(self):
        result = []
        for e in self._compare_files:
            entry = {'file': e['file'], 'code': e['code'], 'label': e['label']}
            uni_checks = e.get('uni_checks', {})
            if uni_checks:
                sel = [i for i, cb in uni_checks.items() if cb.isChecked()]
                if sel: entry['universes'] = sel
            result.append(entry)
        return result or None

    # ── Auto-detect / reactive UI ─────────────────────────────────────────────

    def _on_input_change(self, path):
        self._autodetect_code(path)
        if path.endswith('.init'):
            self._update_universe_selector(path)

    def _autodetect_code(self, path):
        ext = Path(path).suffix.lower()
        if ext == '.init':    self.w_code.setCurrentText('tao')
        elif ext == '.ele':   self.w_code.setCurrentText('elegant')
        elif ext == '.json':  self.w_code.setCurrentText('xsuite')
        elif ext == '.tfs':   self.w_code.setCurrentText('madx')
        self._update_xsuite_rows()
        self._update_madx_rows()

    def _update_xsuite_rows(self):
        if not hasattr(self, '_xsuite_widget'): return
        if self.w_code.currentText() == 'xsuite':
            self._xsuite_widget.show()
        else:
            self._xsuite_widget.hide()

    def _update_madx_rows(self):
        if not hasattr(self, '_madx_widget'): return
        if self.w_code.currentText() == 'madx':
            self._madx_widget.show()
        else:
            self._madx_widget.hide()

    def _update_universe_selector(self, path):
        # Clear existing checkboxes and label edits
        while self._uni_checks_h.count():
            item = self._uni_checks_h.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._uni_checks = {}
        self._uni_label_edits = {}
        try:
            n, labels = _parse_tao_init(path)
        except Exception:
            self._uni_widget.hide(); return
        self._uni_n = n
        if n <= 1:
            self._uni_widget.hide(); return
        for i in range(1, n + 1):
            lbl = labels.get(i, f'u{i}')
            # Container for checkbox + label edit
            cell = QWidget(); cell.setStyleSheet("background: transparent;")
            cell_h = QHBoxLayout(cell); cell_h.setContentsMargins(0, 0, 0, 0); cell_h.setSpacing(4)
            cb = QCheckBox(f"u{i}"); cb.setChecked(True); cb.setFont(FONT_MAIN)
            cb.setStyleSheet(_CHK_SS); self._uni_checks[i] = cb
            cell_h.addWidget(cb)
            le = QLineEdit(lbl); le.setFixedWidth(100); le.setFont(FONT_MAIN)
            le.setStyleSheet(_ENTRY_SS)
            self._uni_label_edits[i] = le
            cell_h.addWidget(le)
            self._uni_checks_h.addWidget(cell)
        self._uni_checks_h.addStretch()
        self._uni_widget.show()

    def _get_selected_universes(self):
        if not self._uni_checks or self._uni_n <= 1: return None
        sel = [i for i, cb in self._uni_checks.items() if cb.isChecked()]
        return sel if sel else None

    def _get_uni_label_overrides(self):
        if not self._uni_label_edits: return None
        return {i: le.text().strip() for i, le in self._uni_label_edits.items()
                if le.text().strip()}

    _PARTICLE_MASS = {"Electron": 0.511, "Proton": 938.272, "Muon": 105.658}

    def _update_emit_ui(self):
        is_norm = self.w_emit_norm.isChecked()
        self._norm_widget.setVisible(is_norm)
        if is_norm and self.w_particle.currentText() == "Custom":
            self._mass_widget.show()
        else:
            self._mass_widget.hide()
        self._update_betagamma()

    def _update_betagamma(self):
        if not hasattr(self, 'w_emit_norm'): return
        if not self.w_emit_norm.isChecked():
            self.w_betagamma.setText("—"); return
        try:
            import math
            E = float(self.w_energy.text().strip())
            m = self._PARTICLE_MASS.get(self.w_particle.currentText())
            if m is None: m = float(self.w_mass.text().strip())
            self.w_betagamma.setText(f"{math.sqrt((E/m)**2 - 1):.4f}")
        except Exception:
            self.w_betagamma.setText("—")

    def _get_font_sizes(self):
        """Collect font size overrides. Returns None if all blank."""
        def _iv(w):
            t = w.text().strip()
            try: return int(t) if t else None
            except: return None
        d = {}
        v = _iv(self.w_fs_axis);   d['axis_label'] = v if v else None
        v = _iv(self.w_fs_tick);   d['tick']       = v if v else None
        v = _iv(self.w_fs_title);  d['title']      = v if v else None
        v = _iv(self.w_fs_annot);  d['annot']      = v if v else None
        v = _iv(self.w_fs_legend); d['legend']     = v if v else None
        d = {k: v for k, v in d.items() if v is not None}
        return d if d else None

    def _get_geometric_emittances(self):
        def _p(widget):
            t = widget.text().strip()
            if not t: return None
            try: return float(t)
            except: return None
        ex = _p(self.w_emitx); ey = _p(self.w_emity)
        if self.w_emit_norm.isChecked():
            try:
                bg = float(self.w_betagamma.text())
                if ex is not None: ex /= bg
                if ey is not None: ey /= bg
            except: pass
        return ex, ey

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_input(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select input file", "",
            "All supported (*.init *.ele *.json *.tfs);;Tao init (*.init);;ELEGANT ele (*.ele);;xsuite JSON (*.json);;MAD-X TFS (*.tfs);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog)
        if f: self.w_input.setText(f)

    def _browse_madx_survey(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select MAD-X survey file", "",
            "TFS files (*.tfs);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog)
        if f: self.w_madx_survey.setText(f)

    def _browse_tunnel(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select tunnel wall file", "",
            "Data files (*.dat *.txt *.csv);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog)
        if f: self.w_tunnel_file.setText(f)

    def _browse_output(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Save output HTML", "optics.html",
            "HTML files (*.html);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog)
        if f: self.w_output.setText(f)

    # ── Collect kwargs ────────────────────────────────────────────────────────

    def _collect_kwargs(self):
        inp = self.w_input.text().strip()
        if not inp or inp == "tao.init / run.ele / lattice.json":
            raise ValueError("Please select an input file.")

        def _fn(widget):
            t = widget.text().strip()
            if not t: return None
            try: return float(t)
            except: raise ValueError(f"Invalid number: '{t}'")

        rng = self.w_range.text().strip() or None
        ttl = self.w_title.text().strip() or None
        fp_xz = self.w_fp_xz_range.text().strip() or None
        fp_yz = self.w_fp_yz_range.text().strip() or None
        xsl   = self.w_xsuite_line.text().strip() or None
        ex, ey = self._get_geometric_emittances()

        return dict(
            input_file=inp, code=self.w_code.currentText(),
            output_file=self.w_output.text().strip() or "optics.html",
            show_element_labels=not self.w_no_labels.isChecked(), show=False,
            save_png=self.w_png.isChecked(), save_pdf=self.w_pdf.isChecked(),
            csv_base=self.w_csv_base.text().strip() or 'lattice',
            dpi=int(self.w_dpi.text().strip() or "300"),
            flip_bend=self.w_flip_bend.isChecked(),
            element_height_xz=float(self.w_elem_h.text().strip() or "0.05"),
            element_height_yz=_fn(self.w_elem_h_yz),
            fp_xz_range=fp_xz, fp_yz_range=fp_yz,
            panels=self._get_panels(), layout=self.w_layout.currentText(), srange=rng,
            panel_annotations=self._get_panel_annotations(),
            legend_positions=self._get_legend_positions(),
            font_sizes=self._get_font_sizes(),
            panel_heights=self._get_panel_heights(),
            emit_x=ex, emit_y=ey, sigma_dp=_fn(self.w_sigmadp),
            n_sigma=_fn(self.w_nsigma) or 1.0, title=ttl,
            dark_mode=self.w_dark.isChecked(), aspect_ratio=self.w_aspect.text().strip() or None,
            legend_inside=self.w_legend_inside.isChecked(),
            xsuite_twiss=self.w_xsuite_twiss.currentText(), xsuite_line=xsl,
            universes=self._get_selected_universes(),
            uni_label_overrides=self._get_uni_label_overrides(),
            madx_survey=self.w_madx_survey.text().strip() or None,
            show_tune=self.w_show_tune.isChecked(),
            show_tunnel=self.w_show_tunnel.isChecked(),
            tunnel_wall_file=self.w_tunnel_file.text().strip() or None,
            show_markers=self.w_show_markers.isChecked(),
            show_markers_bar=self.w_show_markers_bar.isChecked(),
            bar_lite=self.w_bar_lite.isChecked(),
            show_floor=False,  # floor plan now handled as panel type
            color_beampipes=self.w_color_beampipes.isChecked(),
            show_xz=self.w_show_xz.isChecked(),
            show_yz=self.w_show_yz.isChecked(),
            show_titles=self.w_show_titles.isChecked(),
            panel_spacing=float(self.w_panel_spacing.text().strip() or '80'),
            compare=self._get_compare_list(),
            compare_mode=self.w_compare_mode.currentText().lower().replace(' ', '').replace('(%)', '%'),
            normalize_s=self.w_normalize_s.isChecked(),
        )

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return

        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self.open_btn.setStyleSheet("")  # reset green highlight from previous run
        self._set_status("Running…"); self._progress.setValue(0); self._pct_lbl.setText("0%")
        self._log("\n" + "─" * 60 + "\n", "dim")
        self._log(f"▶ code={kwargs['code']}  layout={kwargs['layout']}  panels={kwargs['panels']}\n", "info")
        self._log("─" * 60 + "\n", "dim")

        self._cancelled = False

        def _worker():
            try:
                kwargs['log_fn']      = lambda m: self._sig_log.emit(m, _clf(m))
                kwargs['progress_fn'] = lambda p, l: self._sig_progress.emit(int(p), l or "")
                plot_optics(**kwargs)
                out = str(Path(kwargs['output_file']).resolve())
                self._last_output = out
                self._sig_log.emit(f"\n✓ Done — {out}\n", "ok")
                self._sig_progress.emit(100, "Done ✓")
                self._sig_done.emit(kwargs['input_file'])
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ Error:\n{tb}\n", "error")
                self._sig_progress.emit(0, "Failed")
                self._sig_failed.emit("")
            finally:
                self._sig_finally.emit()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _cancel(self):
        self._log("\n[Cancel requested.]\n", "warn")
        self._set_status("Cancelling…")

    def _open_plot(self):
        import webbrowser
        if self._last_output and Path(self._last_output).exists():
            webbrowser.open(f"file://{self._last_output}")
        else:
            QMessageBox.warning(self, "Open Plot", "Output file not found. Run first.")

    def _set_progress(self, pct, label=""):
        self._progress.setValue(int(pct))
        self._pct_lbl.setText(f"{int(pct)}%" if pct > 0 else "")
        if label: self._set_status(label)

    def _on_run_done(self, input_file):
        self.open_btn.setEnabled(True)
        self.open_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SUCCESS}; border: 1px solid {SUCCESS};
                border-radius: 8px; color: {CRUST}; font-weight: bold;
            }}
            QPushButton:hover {{ background: {SUCCESS}; color: {CRUST}; opacity: 0.9; }}
        """)
        self._save_recent(input_file)

    def _on_run_failed(self, _unused):
        pass   # status already set via _sig_progress

    def _on_run_finally(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.dryrun_btn.setEnabled(True)
        self._progress.setValue(0)
        self._pct_lbl.setText("")

    def _progress_safe(self, pct, label=None):
        """Thread-safe: emit signal, Qt delivers it to main thread."""
        self._sig_progress.emit(int(pct), label or "")

    def _dry_run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return

        self.run_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self._set_status("Inspecting…")
        self._log("\n🔍 Dry run — loading lattice only…\n", "info")

        def _worker():
            try:
                code = kwargs['code']; inp = kwargs['input_file']
                log = lambda m: self._sig_log.emit(m, "info")
                if code == 'tao':     data = load_tao(inp, log_fn=log)
                elif code == 'xsuite': data = load_xsuite(inp, log_fn=log)
                else:                  data = load_elegant(inp, log_fn=log)
                s = data['s']; elems = data['elements']
                msg = f"\n✓ {len(elems)} elements, s = {float(s[0]):.3f} → {float(s[-1]):.3f} m\n"
                self._sig_log.emit(msg, "ok")
                self._sig_progress.emit(0, "Inspection done ✓")
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ Error:\n{tb}\n", "error")
                self._sig_progress.emit(0, "Failed")
            finally:
                self._sig_finally.emit()

        threading.Thread(target=_worker, daemon=True).start()

    def _export_csv(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return
        # Use the base name from the field, trigger via save_csv flag
        kwargs['save_csv'] = True
        kwargs['csv_base'] = self.w_csv_base.text().strip() or 'lattice'
        self._set_status("Exporting CSV…")
        def _worker():
            try:
                plot_optics(**kwargs)
                self._sig_progress.emit(0, "CSV exported ✓")
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ CSV error:\n{tb}\n", "error")
        threading.Thread(target=_worker, daemon=True).start()

    def _copy_path(self):
        if self._last_output:
            QApplication.clipboard().setText(self._last_output)
            self._set_status("Path copied ✓")
        else:
            QMessageBox.warning(self, "Copy Path", "No output yet. Run first.")

    # ── Log ───────────────────────────────────────────────────────────────────

    _LOG_COLORS = {"ok": SUCCESS, "warn": WARN, "error": ERROR, "dim": FG_DIM, "info": FG}

    def _write_log(self, text, tag):
        """Write text directly to the log widget with color."""
        color = self._LOG_COLORS.get(tag, FG)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFont(FONT_MONO)
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        if self._log_autoscroll:
            self.log.ensureCursorVisible()

    def _log(self, text, tag="info"):
        import time as _time
        # ── Deduplication ─────────────────────────────────────────────────────
        stripped = text.strip()
        if stripped and stripped == self._log_last_line.strip():
            self._log_repeat_count += 1
            # Update the last line in widget to show repeat count
            cursor = self.log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.KeepAnchor)
            color = self._LOG_COLORS.get(tag, FG)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(FG_DIM))
            fmt.setFont(FONT_MONO)
            cursor.setCharFormat(fmt)
            cursor.insertText(f"  ↑ repeated {self._log_repeat_count + 1}×\n")
            self.log.setTextCursor(cursor)
            if self._log_autoscroll: self.log.ensureCursorVisible()
            return
        else:
            self._log_repeat_count = 0
            self._log_last_line = stripped

        # ── Timestamp ─────────────────────────────────────────────────────────
        ts = _time.strftime("%H:%M:%S")
        display = f"[{ts}] {text}" if text.strip() else text

        # ── Store in full history ──────────────────────────────────────────────
        self._log_full.append((display, tag))

        # ── Apply filter ──────────────────────────────────────────────────────
        if self._log_filter == 'error' and tag not in ('error',):
            return
        if self._log_filter == 'warn' and tag not in ('error', 'warn'):
            return

        self._write_log(display, tag)

    def _log_safe(self, text, tag="info"):
        """Thread-safe: emit signal, Qt delivers it to main thread."""
        self._sig_log.emit(text, tag)

    def _clear_log(self):
        self.log.clear()
        self._log_full = []
        self._log_last_line = ''
        self._log_repeat_count = 0
        self._log("Log cleared.\n", "dim")

    # ── Recent files ──────────────────────────────────────────────────────────

    def _load_recent(self):
        try:
            import json
            data = json.loads(self._RECENT_FILE.read_text())
            return [p for p in data if Path(p).exists()]
        except: return []

    def _save_recent(self, path):
        import json
        recent = [p for p in self._load_recent() if p != path]
        recent.insert(0, path)
        try: self._RECENT_FILE.write_text(json.dumps(recent[:self._MAX_RECENT]))
        except: pass
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        if not hasattr(self, '_recent_menu'): return
        self._recent_menu.clear()
        recent = self._load_recent()
        if not recent:
            a = QAction("(no recent files)", self); a.setEnabled(False)
            self._recent_menu.addAction(a); return
        for p in recent:
            label = Path(p).name + "  —  " + str(Path(p).parent)
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, f=p: self.w_input.setText(f))
            self._recent_menu.addAction(act)

    # ── Presets ───────────────────────────────────────────────────────────────

    def _load_presets(self):
        try:
            import json; return json.loads(self._PRESET_FILE.read_text())
        except: return {}

    def _save_presets(self, presets):
        import json
        try: self._PRESET_FILE.write_text(json.dumps(presets, indent=2))
        except Exception as e: QMessageBox.critical(self, "Preset Error", str(e))

    def _collect_preset(self):
        return {
            'code':       self.w_code.currentText(),
            'output':     self.w_output.text(),
            'layout':     self.w_layout.currentText(),
            'range':      self.w_range.text(),
            'panels':     [{'name': p['name'], 'spec': p['spec'], 'annot_pattern': p.get('annot_pattern','')} for p in self._panels],
            'emit_type':  'normalized' if self.w_emit_norm.isChecked() else 'geometric',
            'emitx':      self.w_emitx.text(),     'emity':    self.w_emity.text(),
            'sigmadp':    self.w_sigmadp.text(),   'nsigma':   self.w_nsigma.text(),
            'particle':   self.w_particle.currentText(), 'energy': self.w_energy.text(),
            'title':      self.w_title.text(),     'elem_h':   self.w_elem_h.text(),
            'elem_h_yz':  self.w_elem_h_yz.text(), 'fp_xz_range': self.w_fp_xz_range.text(),
            'fp_yz_range': self.w_fp_yz_range.text(),
            'no_labels':  self.w_no_labels.isChecked(), 'flip_bend': self.w_flip_bend.isChecked(),
            'dark_mode':  self.w_dark.isChecked(), 'png': self.w_png.isChecked(),
            'pdf':        self.w_pdf.isChecked(),  'dpi': self.w_dpi.text(),
            'csv_base': self.w_csv_base.text().strip(),
            'aspect':     self.w_aspect.text(),    'legend_inside': self.w_legend_inside.isChecked(),
            'compare_files': list(self._compare_files),
            'compare_mode':  self.w_compare_mode.currentText(),
            'normalize_s':   self.w_normalize_s.isChecked(),
            'color_beampipes': self.w_color_beampipes.isChecked(),
            'show_markers':     self.w_show_markers.isChecked(),
            'show_markers_bar': self.w_show_markers_bar.isChecked(),
            'bar_lite':         self.w_bar_lite.isChecked(),
            'fs_axis':  self.w_fs_axis.text(),  'fs_tick':  self.w_fs_tick.text(),
            'fs_title': self.w_fs_title.text(), 'fs_annot': self.w_fs_annot.text(),
            'fs_legend':self.w_fs_legend.text(),
            'show_xz':         self.w_show_xz.isChecked(),
            'show_yz':         self.w_show_yz.isChecked(),
            'show_titles':     self.w_show_titles.isChecked(),
            'panel_spacing':   self.w_panel_spacing.text().strip(),
            'madx_survey':     self.w_madx_survey.text().strip(),
        }

    def _apply_preset(self, data):
        def _st(widget, key):
            if key in data and hasattr(widget, 'setText'): widget.setText(str(data[key]))
        def _sc(widget, key):
            if key in data and hasattr(widget, 'setChecked'): widget.setChecked(bool(data[key]))
        def _sct(widget, key):
            if key in data and hasattr(widget, 'setCurrentText'): widget.setCurrentText(str(data[key]))

        _sct(self.w_code,     'code');     _st(self.w_output, 'output')
        _sct(self.w_layout,   'layout');   _st(self.w_range,  'range')
        _st(self.w_emitx,  'emitx');       _st(self.w_emity,  'emity')
        _st(self.w_sigmadp,'sigmadp');     _st(self.w_nsigma, 'nsigma')
        _sct(self.w_particle,'particle');  _st(self.w_energy, 'energy')
        _st(self.w_title,  'title');       _st(self.w_elem_h, 'elem_h')
        _st(self.w_elem_h_yz,'elem_h_yz'); _st(self.w_fp_xz_range,'fp_xz_range')
        _st(self.w_fp_yz_range,'fp_yz_range')
        _sc(self.w_no_labels,'no_labels'); _sc(self.w_flip_bend,'flip_bend')
        _sc(self.w_dark,'dark_mode');      _sc(self.w_png,'png');  _sc(self.w_pdf,'pdf')
        _st(self.w_dpi,'dpi');             _st(self.w_aspect,'aspect')
        _st(self.w_csv_base, 'csv_base')
        _sc(self.w_legend_inside,'legend_inside')
        _sct(self.w_compare_mode, 'compare_mode')
        _sc(self.w_normalize_s, 'normalize_s')
        _sc(self.w_color_beampipes, 'color_beampipes')
        _sc(self.w_show_markers,     'show_markers')
        _sc(self.w_show_markers_bar, 'show_markers_bar')
        _sc(self.w_bar_lite,         'bar_lite')
        _st(self.w_fs_axis,  'fs_axis');  _st(self.w_fs_tick,  'fs_tick')
        _st(self.w_fs_title, 'fs_title'); _st(self.w_fs_annot, 'fs_annot')
        _st(self.w_fs_legend,'fs_legend')
        _sc(self.w_show_xz, 'show_xz')
        _sc(self.w_show_yz, 'show_yz')
        _sc(self.w_show_titles, 'show_titles')
        _st(self.w_panel_spacing, 'panel_spacing')
        if 'madx_survey' in data: self.w_madx_survey.setText(str(data.get('madx_survey', '')))

        if 'emit_type' in data:
            is_norm = str(data['emit_type']).lower() == 'normalized'
            self.w_emit_norm.setChecked(is_norm)
            self.w_emit_geo.setChecked(not is_norm)

        if 'panels' in data:
            loaded = [{'name': p.get('name', 'Panel'), 'spec': p.get('spec', 'twiss'),
                       'annot_pattern': p.get('annot_pattern', '')}
                      for p in data['panels']]
            if loaded: self._panels = loaded; self._render_panel_list()

        if 'compare_files' in data:
            self._compare_files = list(data['compare_files'])
            self._render_compare_list()

        self._update_emit_ui()

    def _preset_save_dialog(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name: return
        presets = self._load_presets(); presets[name] = self._collect_preset()
        self._save_presets(presets); self._refresh_preset_menu()
        self._log(f"[preset] Saved '{name}'\n", "ok")

    def _preset_delete_dialog(self):
        presets = self._load_presets()
        if not presets:
            QMessageBox.information(self, "Delete Preset", "No saved presets."); return
        name, ok = QInputDialog.getText(self, "Delete Preset",
            "Preset to delete:\n" + ", ".join(presets.keys()))
        if ok and name and name in presets:
            del presets[name]; self._save_presets(presets); self._refresh_preset_menu()
            self._log(f"[preset] Deleted '{name}'\n", "warn")

    def _refresh_preset_menu(self):
        if not hasattr(self, '_preset_menu'): return
        self._preset_menu.clear()
        presets = self._load_presets()
        if not presets:
            a = QAction("(no saved presets)", self); a.setEnabled(False)
            self._preset_menu.addAction(a); return
        for name in presets:
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, n=name:
                self._apply_preset(self._load_presets().get(n, {})))
            self._preset_menu.addAction(act)
        self._preset_menu.addSeparator()
        self._preset_menu.addAction(QAction("Delete a preset…", self,
                                            triggered=self._preset_delete_dialog))

# ═══════════════════════════════════════════════════════════════════════════════
#  Overlay compositors — CustomPanelOverlay, ExprPanelOverlay
# ═══════════════════════════════════════════════════════════════════════════════