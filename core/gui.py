# =============================================================================
# core/gui.py — RanOptics main GUI (RanOpticsGUI)
# =============================================================================

from __future__ import annotations
import fnmatch, json, math, os, re, threading, time, traceback
from pathlib import Path
import numpy as np

from PySide6.QtCore    import Qt, Signal, QTimer, QSize
from PySide6.QtGui     import QAction, QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QColorDialog,
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTabWidget, QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

import core.themes as _th
from core.themes import (
    FONT_BOLD, FONT_HDR, FONT_MAIN, FONT_MONO, FONT_SEC, FONT_SMALL,
)
from core.utils import (
    _clf, _make_scroll_widget, check_backend_ready,
    _sec, _card, _row, _lbl, _ent, _btn, _chk, _dd, _hint, _help, _rb,
    _parse_yrange, _parse_fp_range,
)
from core.engine import plot_optics
from core.loaders import load_tao, load_elegant, load_xsuite, load_madx, _parse_tao_init
from core.overlays import (
    CustomPanelOverlay, ExprPanelOverlay,
    _TAO_DATA_CATEGORIES, _ELEGANT_TWI_COLUMNS, _ELEGANT_CEN_COLUMNS,
    _ELEGANT_SIG_COLUMNS, _ELEGANT_TWI_SCALARS,
)

# ── Main GUI class ────────────────────────────────────────────────────────────

class RanOpticsGUI(QMainWindow):
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
            QMainWindow {{ background: {_th.BG}; }}
            QWidget {{ background: {_th.BG}; color: {_th.FG}; }}
            QMenuBar {{ background: {_th.MANTLE}; color: {_th.FG}; border-bottom: 1px solid {_th.BORDER}; }}
            QMenuBar::item:selected {{ background: {_th.SURFACE2}; color: {_th.ACCENT}; }}
            QMenu {{ background: {_th.PANEL}; color: {_th.FG}; border: 1px solid {_th.BORDER}; border-radius: 6px; padding: 4px; }}
            QMenu::item:selected {{ background: {_th.SURFACE2}; color: {_th.ACCENT}; }}
            QStatusBar {{ background: {_th.MANTLE}; color: {_th.FG_DIM}; border-top: 1px solid {_th.BORDER}; }}
            QToolTip {{
                background: {_th.PANEL}; color: {_th.FG};
                border: 1px solid {_th.BORDER}; border-radius: 6px;
                padding: 4px 8px;
            }}
        """)

        self._last_output = None
        self._uni_checks  = {}   # {i: QCheckBox}
        self._uni_label_edits = {}  # {i: QLineEdit}
        self._uni_n       = 1
        self._panels      = [{'name': 'Floor Plan X-Z',      'spec': 'floor-xz'},
                              {'name': 'Twiss & Dispersion', 'spec': 'twiss'},
                              {'name': 'Beamline Bar',       'spec': 'bar'}]
        self._panel_rows  = []
        # Element color overrides — persisted in presets
        self._elem_colors = {}
        # Color swatch buttons — keyed by element type key
        self._color_btns  = {}
        # Grid layout state
        self._grid_enabled = False
        self._grid_rows    = 2
        self._grid_cols    = 2

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
        QTimer.singleShot(0, self._restyle)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{
                background: {_th.CRUST}; color: {_th.FG_LBL};
                border-bottom: 1px solid {_th.BORDER};
            }}
            QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
            QMenuBar::item:selected {{ background: {_th.SURFACE2}; color: {_th.FG}; }}
            QMenu {{
                background: {_th.MANTLE}; color: {_th.FG};
                border: 1px solid {_th.BORDER}; border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {_th.PANEL}; color: {_th.ACCENT}; }}
            QMenu::separator {{ background: {_th.BORDER}; height: 1px; margin: 4px 8px; }}
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
        self._header_widget = QWidget(); self._header_widget.setFixedHeight(80)
        self._header_widget.setStyleSheet(f"background: {_th.PANEL}; border-bottom: 1px solid {_th.BORDER};")
        row = QHBoxLayout(self._header_widget); row.setContentsMargins(22, 0, 22, 0); row.setSpacing(14)

        # Logo icon (rounded box with sparkline SVG) + wordmark
        from pathlib import Path as _Path
        _logo_dir = _Path(__file__).resolve().parent.parent / "logo"
        self._logo_dir = _logo_dir

        self._logo_lbl = QLabel()
        self._logo_lbl.setFixedSize(44, 44)
        self._logo_lbl.setAlignment(Qt.AlignCenter)
        self._logo_lbl.setStyleSheet("background: transparent;")
        row.addWidget(self._logo_lbl)

        txt = QWidget(); txt.setStyleSheet("background: transparent;")
        tv  = QVBoxLayout(txt); tv.setContentsMargins(0, 0, 0, 0); tv.setSpacing(4)
        name_row = QWidget(); name_row.setStyleSheet("background: transparent;")
        nr = QHBoxLayout(name_row); nr.setContentsMargins(0,0,0,0); nr.setSpacing(0)
        self._hdr_ran = QLabel("Ran"); self._hdr_ran.setFont(FONT_HDR)
        self._hdr_ran.setStyleSheet(f"color: {_th.ACCENT}; background: transparent;")
        self._hdr_opt = QLabel("Optics"); self._hdr_opt.setFont(FONT_HDR)
        self._hdr_opt.setStyleSheet(f"color: {_th.FG}; background: transparent;")
        nr.addWidget(self._hdr_ran); nr.addWidget(self._hdr_opt); nr.addStretch()
        tv.addWidget(name_row)
        self._hdr_sub = QLabel("Accelerator Optics Plotter  ·  v2.0.0"); self._hdr_sub.setFont(FONT_MONO)
        self._hdr_sub.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent; border-bottom: 1px solid {_th.COPPER}; padding-bottom: 1px;")
        tv.addWidget(self._hdr_sub)
        row.addWidget(txt)
        row.addStretch()

        # Right side: author/support on top, theme toggle below — all right-aligned
        rf = QWidget(); rf.setStyleSheet("background: transparent;")
        rv = QVBoxLayout(rf); rv.setContentsMargins(0,0,0,0); rv.setSpacing(6)
        rv.setAlignment(Qt.AlignRight)

        # Author + support
        info_w = QWidget(); info_w.setStyleSheet("background: transparent;")
        info_v = QVBoxLayout(info_w); info_v.setContentsMargins(0,0,0,0); info_v.setSpacing(2)
        self._author_lbl = QLabel()
        self._author_lbl.setText(
            f'<span style="color:{_th.FG_DIM}; font-size:11px;">Randika Gamage · '
            f'<a href="mailto:randika@jlab.org" style="color:{_th.FG_DIM}; text-decoration:none;">'
            f'randika@jlab.org</a></span>'
        )
        self._author_lbl.setFont(FONT_SMALL); self._author_lbl.setOpenExternalLinks(True)
        self._author_lbl.setTextFormat(Qt.RichText); self._author_lbl.setAlignment(Qt.AlignRight)
        self._support_lbl = QLabel("Good luck, I believe in you")
        self._support_lbl.setFont(FONT_SMALL); self._support_lbl.setAlignment(Qt.AlignRight)
        self._support_lbl.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent;")
        info_v.addWidget(self._author_lbl); info_v.addWidget(self._support_lbl)
        rv.addWidget(info_w)

        # Theme toggle
        self._tog_widget = QWidget(); self._tog_widget.setStyleSheet(f"background: {_th.MANTLE}; border: 1px solid {_th.BORDER}; border-radius: 8px;")
        tl = QHBoxLayout(self._tog_widget); tl.setContentsMargins(3, 3, 3, 3); tl.setSpacing(2)
        self._btn_light = QPushButton("☀ Light"); self._btn_light.setFont(FONT_SMALL)
        self._btn_light.setFixedHeight(26); self._btn_light.setCheckable(True)
        self._btn_dark  = QPushButton("☾ Dark");  self._btn_dark.setFont(FONT_SMALL)
        self._btn_dark.setFixedHeight(26);  self._btn_dark.setCheckable(True)
        self._btn_dark.setChecked(True)
        for b in (self._btn_light, self._btn_dark):
            b.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: 1px solid transparent;
                    border-radius: 6px; color: {_th.FG_DIM}; padding: 4px 12px;
                }}
                QPushButton:checked {{
                    background: {_th.ASOFT}; border-color: {_th.ACCENT}; color: {_th.ACCENT};
                }}
                QPushButton:hover:!checked {{ background: {_th.SURFACE2}; color: {_th.FG}; }}
            """)
        self._btn_light.clicked.connect(lambda: self._switch_theme("light"))
        self._btn_dark.clicked.connect(lambda: self._switch_theme("dark"))
        tl.addWidget(self._btn_light); tl.addWidget(self._btn_dark)
        tog_wrap = QWidget(); tog_wrap.setStyleSheet("background: transparent;")
        tw = QHBoxLayout(tog_wrap); tw.setContentsMargins(0,0,0,0)
        tw.addStretch(); tw.addWidget(self._tog_widget)
        rv.addWidget(tog_wrap)

        row.addWidget(rf)
        self._root_layout.addWidget(self._header_widget)

    def _switch_theme(self, mode):
        from core import themes as _th
        _th.apply_theme(mode)
        th = _th  # shorthand
        # Update QApplication global stylesheet — covers most widgets
        from PySide6.QtWidgets import QApplication
        QApplication.instance().setStyleSheet(f"""
            QWidget {{ background-color: {th.BG}; color: {th.FG}; }}
            QMainWindow {{ background-color: {th.BG}; }}
            QMenuBar {{ background-color: {th.MANTLE}; color: {th.FG}; border-bottom: 1px solid {th.BORDER}; }}
            QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
            QMenuBar::item:selected {{ background: {th.SURFACE2}; color: {th.ACCENT}; border-radius: 4px; }}
            QMenu {{ background: {th.PANEL}; color: {th.FG}; border: 1px solid {th.BORDER}; border-radius: 6px; padding: 4px; }}
            QMenu::item:selected {{ background: {th.SURFACE2}; color: {th.ACCENT}; border-radius: 4px; }}
            QStatusBar {{ background: {th.MANTLE}; color: {th.FG_DIM}; border-top: 1px solid {th.BORDER}; }}
            QToolTip {{ background: {th.PANEL}; color: {th.FG}; border: 1px solid {th.BORDER}; border-radius: 6px; padding: 4px 8px; }}
            QLineEdit {{ background: {th.MANTLE}; border: 1px solid {th.BORDER}; border-radius: 7px; color: {th.FG}; padding: 6px 10px; }}
            QLineEdit:focus {{ border-color: {th.ACCENT}; border-left: 3px solid {th.ACCENT}; background: {th.BG}; }}
            QLineEdit[readOnly="true"] {{ color: {th.FG_DIM}; background: {th.PANEL}; }}
            QComboBox {{ background: {th.MANTLE}; border: 1px solid {th.BORDER}; border-radius: 7px; color: {th.FG}; padding: 5px 10px; }}
            QComboBox:focus {{ border-color: {th.ACCENT}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{ background: {th.PANEL}; color: {th.FG}; border: 1px solid {th.BORDER}; border-radius: 6px; selection-background-color: {th.ACCENT}; selection-color: {th.AINK}; }}
            QPushButton {{ background: {th.PANEL}; border: 1px solid {th.BORDER}; border-radius: 7px; color: {th.ACCENT}; padding: 5px 10px; }}
            QPushButton:hover {{ background: {th.SURFACE2}; border-color: {th.ACCENT}; }}
            QPushButton:pressed {{ background: {th.BORDER}; }}
            QPushButton:disabled {{ color: {th.FG_DIM}; border-color: {th.BORDER}; }}
            QCheckBox {{ color: {th.FG}; spacing: 7px; }}
            QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid {th.SURFACE2}; background: {th.MANTLE}; }}
            QCheckBox::indicator:checked {{ background: {th.ACCENT}; border-color: {th.ACCENT}; }}
            QTabWidget::pane {{ background: {th.PANEL}; border: 1px solid {th.BORDER}; border-radius: 9px; top: -1px; }}
            QTabBar::tab {{ background: {th.MANTLE}; color: {th.FG_LBL}; padding: 7px 18px; border: 1px solid {th.BORDER}; border-bottom: none; border-top-left-radius: 7px; border-top-right-radius: 7px; font-weight: 500; }}
            QTabBar::tab:selected {{ background: {th.PANEL}; color: {th.ACCENT}; border-bottom-color: {th.PANEL}; }}
            QTabBar::tab:hover {{ background: {th.SURFACE2}; color: {th.FG}; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: {th.MANTLE}; width: 6px; margin: 0; border-radius: 3px; }}
            QScrollBar::handle:vertical {{ background: {th.SURFACE2}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background: {th.ACCENT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QTextEdit {{ background: {th.MANTLE}; color: {th.FG}; border: 1px solid {th.BORDER}; border-radius: 7px; }}
            QLabel {{ background: transparent; color: {th.FG}; }}
            QSplitter::handle {{ background: {th.BORDER}; }}
        """)
        # Update toggle button states
        self._btn_light.setChecked(mode == 'light')
        self._btn_dark.setChecked(mode == 'dark')
        self._restyle()

    def _restyle(self):
        """Re-apply all inline stylesheets using live theme values after a theme switch."""
        import core.themes as th
        # Stylesheet helpers rebuilt by apply_theme — grab fresh copies
        _entry = th._ENTRY_SS; _combo = th._COMBO_SS; _chk = th._CHK_SS

        # ── Menu bar ──────────────────────────────────────────────────────────
        self.menuBar().setStyleSheet(f"""
            QMenuBar {{
                background: {th.CRUST}; color: {th.FG_LBL};
                border-bottom: 1px solid {th.BORDER};
            }}
            QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
            QMenuBar::item:selected {{ background: {th.SURFACE2}; color: {th.FG}; }}
            QMenu {{
                background: {th.MANTLE}; color: {th.FG};
                border: 1px solid {th.BORDER}; border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {th.PANEL}; color: {th.ACCENT}; }}
        """)

        # ── Header ────────────────────────────────────────────────────────────
        self._header_widget.setStyleSheet(f"background: {th.PANEL}; border-bottom: 1px solid {th.BORDER};")
        _is_dark = (th._current_mode == 'dark')
        from PySide6.QtGui import QPixmap
        _px_file = "ranoptics_mark_light.png" if _is_dark else "ranoptics_mark.png"
        _px = QPixmap(str(self._logo_dir / _px_file))
        self._logo_lbl.setPixmap(_px.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._hdr_ran.setStyleSheet(f"color: {th.ACCENT}; background: transparent;")
        self._hdr_opt.setStyleSheet(f"color: {th.FG}; background: transparent;")
        self._hdr_sub.setStyleSheet(f"color: {th.FG_DIM}; background: transparent; border-bottom: 1px solid {th.COPPER}; padding-bottom: 1px;")
        self._author_lbl.setText(
            f'<span style="color:{th.FG_DIM}; font-size:11px;">Randika Gamage · '
            f'<a href="mailto:randika@jlab.org" style="color:{th.FG_DIM}; text-decoration:none;">'
            f'randika@jlab.org</a></span>'
        )
        self._support_lbl.setStyleSheet(f"color: {th.FG_DIM}; background: transparent;")
        self._tog_widget.setStyleSheet(f"background: {th.MANTLE}; border: 1px solid {th.BORDER}; border-radius: 8px;")
        _tog_ss = f"""
            QPushButton {{
                background: transparent; border: 1px solid transparent;
                border-radius: 6px; color: {th.FG_DIM}; padding: 4px 12px;
            }}
            QPushButton:checked {{
                background: {th.ASOFT}; border-color: {th.ACCENT}; color: {th.ACCENT};
            }}
            QPushButton:hover:!checked {{ background: {th.SURFACE2}; color: {th.FG}; }}
        """
        self._btn_light.setStyleSheet(_tog_ss)
        self._btn_dark.setStyleSheet(_tog_ss)

        # ── Statusbar ─────────────────────────────────────────────────────────
        self._statusbar_w.setStyleSheet(f"background: {th.MANTLE}; border-top: 1px solid {th.BORDER};")
        self._status_lbl.setStyleSheet(f"color: {th.FG_DIM}; background: transparent;")
        self._pct_lbl.setStyleSheet(f"color: {th.ACCENT}; background: transparent;")
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {th.CRUST}; border-radius: 3px; border: none; }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {th.ACCENT}, stop:1 {th.HIGHLIGHT});
                border-radius: 3px;
            }}
        """)

        # ── Run bar ───────────────────────────────────────────────────────────
        self._run_bar.setStyleSheet(f"background: {th.PANEL}; border-top: 1px solid {th.BORDER};")
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {th.ACCENT}; border-radius: 8px;
                color: {th.CRUST}; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {th.HIGHLIGHT}; color: {th.CRUST}; }}
            QPushButton:disabled {{ background: {th.BORDER}; color: {th.FG_DIM}; border: none; }}
        """)
        _abtn_ss = lambda color: f"""
            QPushButton {{
                background: {th.PANEL}; border: 1px solid {color};
                border-radius: 8px; color: {color}; font-weight: 500;
            }}
            QPushButton:hover {{ background: {color}; color: {th.CRUST}; }}
            QPushButton:disabled {{ color: {th.FG_DIM}; border-color: {th.BORDER}; background: {th.PANEL}; }}
        """
        self.stop_btn.setStyleSheet(_abtn_ss(th.ERROR))
        self.open_btn.setStyleSheet(_abtn_ss(th.SUCCESS))
        self.dryrun_btn.setStyleSheet(_abtn_ss(th.COPPER))
        _ghost_ss = f"""
            QPushButton {{
                background: {th.MANTLE}; border: 1px solid {th.BORDER};
                border-radius: 8px; color: {th.FG_LBL};
            }}
            QPushButton:hover {{ background: {th.SURFACE2}; color: {th.FG}; border-color: {th.ACCENT}; }}
        """
        self.csv_btn.setStyleSheet(_ghost_ss)
        self._clr_btn.setStyleSheet(_ghost_ss)

        # ── Log ───────────────────────────────────────────────────────────────
        self._log_frame.setStyleSheet(f"background: {th.BG};")
        self._log_hdr.setStyleSheet(f"color: {th.ACCENT}; background: transparent;")
        self._log_filter_dd.setStyleSheet(_combo)
        self._log_scroll_btn.setStyleSheet(f"""
            QPushButton {{ background: {th.PANEL}; border: 1px solid {th.BORDER};
                border-radius: 6px; color: {th.FG_DIM}; padding: 2px 6px; }}
            QPushButton:checked {{ color: {th.ACCENT}; border-color: {th.ACCENT}; }}
            QPushButton:hover {{ background: {th.SURFACE2}; }}
        """)
        self._log_copy_btn.setStyleSheet(f"""
            QPushButton {{ background: {th.PANEL}; border: 1px solid {th.BORDER};
                border-radius: 6px; color: {th.FG_DIM}; padding: 2px 6px; }}
            QPushButton:hover {{ background: {th.SURFACE2}; color: {th.FG}; }}
        """)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {th.MANTLE}; color: {th.FG};
                border: 1px solid {th.BORDER}; border-radius: 8px;
                padding: 6px; selection-background-color: {th.ACCENT};
            }}
        """)

        # ── Form structural containers ─────────────────────────────────────────
        self._form_outer.setStyleSheet(f"background: {th.BG};")
        self._setup_w.setStyleSheet(f"QWidget {{ background: {th.PANEL}; border-radius: 9px; }}")
        self._setup_hdr.setStyleSheet(f"background: {th.MANTLE}; border-radius: 0px; border-bottom: 1px solid {th.BORDER};")
        self._lbl_setup.setStyleSheet(f"color: {th.FG}; background: transparent;")
        self._right_w.setStyleSheet(f"background: {th.PANEL};")
        self._tab_strip.setStyleSheet(f"background: {th.PANEL}; border-bottom: 1px solid {th.BORDER};")
        self._tab_stack.setStyleSheet(f"background: {th.PANEL};")
        for page in self._tab_pages:
            page.setStyleSheet(f"background: {th.PANEL};")
        # Redraw tab buttons with live colors
        current_idx = self._tab_stack.currentIndex()
        self._switch_tab(current_idx)

        # ── Scroll areas ──────────────────────────────────────────────────────
        for sa in (self._sa_setup, self._sa_panels, self._sa_appearance, self._sa_export):
            sa.setStyleSheet(th._SCROLL_SS)

        # Rebuild right tab content with live theme colors
        self._rebuild_right_tabs()

    def _rebuild_right_tabs(self):
        """Tear down and rebuild all scroll area content with live theme colors."""
        import core.themes as _th_live
        # Save user-entered values that will be lost when widgets are recreated
        _saved_fs = {}
        for _attr in ('w_fs_axis','w_fs_tick','w_fs_title','w_fs_annot','w_fs_legend'):
            try: _saved_fs[_attr] = getattr(self, _attr).text()
            except: pass
        # Clear widget reference dicts before rebuild to avoid dangling C++ pointers
        # NOTE: _uni_checks and _uni_label_edits are NOT cleared here — they are
        # populated by _rebuild_uni_checks() on file load and must persist across
        # theme switches so the universe dropdowns in panel rows stay populated.
        self._panel_height_edits = {}
        self._panel_leg_checks = {}
        self._panel_leg_x_edits = {}
        self._panel_leg_y_edits = {}
        self._panel_annot_checks = {}
        self._panel_annot_edits = {}
        self._panel_grid_checks = {}
        self._color_btns = {}

        # ── Left panel (Setup) ────────────────────────────────────────────────
        new_inner = QWidget(); new_inner.setStyleSheet("background: transparent;")
        new_vbox = QVBoxLayout(new_inner)
        new_vbox.setContentsMargins(0, 4, 0, 8); new_vbox.setSpacing(0)
        self._setup_layout = new_vbox
        self._input_layout = new_vbox
        self._beam_layout  = new_vbox
        self._build_input_section(new_vbox)
        self._build_beam_section(new_vbox)
        new_vbox.addStretch(1)
        self._sa_setup.setWidget(new_inner)
        self._sa_setup.setStyleSheet(_th_live._SCROLL_SS)

        # ── Right tabs ────────────────────────────────────────────────────────
        for sa, build_fns, alias_fns in [
            (self._sa_panels,
             [self._build_panels_section],
             lambda vbox: setattr(self, '_panels_layout', vbox)),
            (self._sa_appearance,
             [self._build_visual_section, self._build_colors_section],
             lambda vbox: [setattr(self, '_appearance_layout', vbox),
                           setattr(self, '_visual_layout', vbox),
                           setattr(self, '_colors_layout', vbox)]),
            (self._sa_export,
             [self._build_export_section],
             lambda vbox: setattr(self, '_export_layout', vbox)),
        ]:
            new_inner = QWidget(); new_inner.setStyleSheet("background: transparent;")
            new_vbox = QVBoxLayout(new_inner)
            new_vbox.setContentsMargins(0, 4, 0, 8); new_vbox.setSpacing(0)
            alias_fns(new_vbox)
            for fn in build_fns:
                fn(new_vbox)
            new_vbox.addStretch(1)
            sa.setWidget(new_inner)
            sa.setStyleSheet(_th_live._SCROLL_SS)

        # Restore saved font size values
        for _attr, _val in _saved_fs.items():
            try: getattr(self, _attr).setText(_val)
            except: pass

    def _switch_tab(self, idx):
        import core.themes as th
        self._tab_stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._tab_btns):
            if i == idx:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; border: none;
                        border-bottom: 2px solid {th.ACCENT};
                        color: {th.FG}; padding: 14px 18px;
                        font-size: 13px; font-weight: 600;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; border: none;
                        border-bottom: 2px solid transparent;
                        color: {th.FG_DIM}; padding: 14px 18px;
                        font-size: 13px;
                    }}
                    QPushButton:hover {{ color: {th.FG}; }}
                """)

    def _build_statusbar(self):
        self._statusbar_w = QWidget(); self._statusbar_w.setFixedHeight(28)
        self._statusbar_w.setStyleSheet(f"background: {_th.MANTLE}; border-top: 1px solid {_th.BORDER};")
        row = QHBoxLayout(self._statusbar_w); row.setContentsMargins(12, 0, 8, 0); row.setSpacing(8)
        self._status_lbl = QLabel("Idle"); self._status_lbl.setFont(FONT_SMALL)
        self._status_lbl.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent;")
        row.addWidget(self._status_lbl); row.addStretch()
        self._progress = QProgressBar(); self._progress.setFixedWidth(180)
        self._progress.setFixedHeight(6); self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {_th.CRUST}; border-radius: 3px; border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {_th.ACCENT}, stop:1 {_th.HIGHLIGHT});
                border-radius: 3px;
            }}
        """)
        self._pct_lbl = QLabel(""); self._pct_lbl.setFont(FONT_BOLD)
        self._pct_lbl.setFixedWidth(40)
        self._pct_lbl.setStyleSheet(f"color: {_th.ACCENT}; background: transparent;")
        row.addWidget(self._progress); row.addWidget(self._pct_lbl)
        self._root_layout.addWidget(self._statusbar_w)

    def _set_status(self, text):
        self._status_lbl.setText(text)

    # ── Run bar ───────────────────────────────────────────────────────────────

    def _build_run_bar(self):
        self._run_bar = QWidget(); self._run_bar.setFixedHeight(52)
        self._run_bar.setStyleSheet(f"background: {_th.PANEL}; border-top: 1px solid {_th.BORDER};")
        row = QHBoxLayout(self._run_bar); row.setContentsMargins(12, 6, 12, 6); row.setSpacing(6)

        self.run_btn = QPushButton("▶  Run"); self.run_btn.setFont(FONT_BOLD)
        self.run_btn.setFixedSize(100, 36); self.run_btn.clicked.connect(self._run)
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_th.ACCENT}; border-radius: 8px;
                color: {_th.CRUST}; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {_th.HIGHLIGHT}; color: {_th.CRUST}; }}
            QPushButton:disabled {{ background: {_th.BORDER}; color: {_th.FG_DIM}; border: none; }}
        """)
        row.addWidget(self.run_btn)

        def _action_btn(text, cmd, color, width=100):
            b = QPushButton(text); b.setFont(FONT_BOLD)
            b.setFixedSize(width, 36); b.clicked.connect(cmd)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {_th.PANEL}; border: 1px solid {color};
                    border-radius: 8px; color: {color}; font-weight: 500;
                }}
                QPushButton:hover {{ background: {color}; color: {_th.CRUST}; }}
                QPushButton:disabled {{ color: {_th.FG_DIM}; border-color: {_th.BORDER}; background: {_th.PANEL}; }}
            """)
            row.addWidget(b); return b

        self.stop_btn    = _action_btn("■  Cancel",    self._cancel,   _th.ERROR)
        self.open_btn    = _action_btn("◉  Open Plot", self._open_plot, _th.SUCCESS, 130)
        self.dryrun_btn  = _action_btn("◷  Dry Run",   self._dry_run,  _th.COPPER, 115)
        self.stop_btn.setEnabled(False); self.open_btn.setEnabled(False)

        self.csv_btn = QPushButton("⤓  Export CSV"); self.csv_btn.setFont(FONT_MAIN)
        self.csv_btn.setFixedSize(130, 36); self.csv_btn.clicked.connect(self._export_csv)
        self.csv_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_th.MANTLE}; border: 1px solid {_th.BORDER};
                border-radius: 8px; color: {_th.FG_LBL};
            }}
            QPushButton:hover {{ background: {_th.SURFACE2}; color: {_th.FG}; border-color: {_th.ACCENT2}; }}
        """)
        row.addWidget(self.csv_btn)
        row.addStretch()

        self._clr_btn = QPushButton("⊗  Clear log"); self._clr_btn.setFont(FONT_MAIN)
        self._clr_btn.setFixedSize(115, 36); self._clr_btn.clicked.connect(self._clear_log)
        self._clr_btn.setStyleSheet(self.csv_btn.styleSheet())
        row.addWidget(self._clr_btn)

        self._root_layout.addWidget(self._run_bar)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log(self):
        self._log_autoscroll  = True
        self._log_filter      = 'all'   # 'all' | 'warn' | 'error'
        self._log_last_line   = ''      # for deduplication
        self._log_repeat_count = 0
        self._log_full        = []      # list of (text, tag) — full unfiltered history

        self._log_frame = QWidget(); self._log_frame.setStyleSheet(f"background: {_th.BG};")
        lv = QVBoxLayout(self._log_frame); lv.setContentsMargins(12, 4, 12, 4); lv.setSpacing(2)

        # ── Log toolbar ───────────────────────────────────────────────────────
        tb = QWidget(); tb.setStyleSheet("background: transparent;")
        tbh = QHBoxLayout(tb); tbh.setContentsMargins(0, 0, 0, 2); tbh.setSpacing(6)

        self._log_hdr = QLabel("OUTPUT LOG"); self._log_hdr.setFont(FONT_SEC)
        self._log_hdr.setStyleSheet(f"color: {_th.ACCENT2}; background: transparent;")
        tbh.addWidget(self._log_hdr)
        tbh.addStretch()

        # Filter dropdown
        self._log_filter_dd = QComboBox(); self._log_filter_dd.setFont(FONT_SMALL)
        self._log_filter_dd.addItems(["All", "Warnings+", "Errors only"])
        self._log_filter_dd.setFixedWidth(110); self._log_filter_dd.setStyleSheet(_th._COMBO_SS)
        self._log_filter_dd.currentIndexChanged.connect(self._on_log_filter_changed)
        tbh.addWidget(self._log_filter_dd)

        # Auto-scroll toggle
        self._log_scroll_btn = QPushButton("⇩ Auto"); self._log_scroll_btn.setFont(FONT_SMALL)
        self._log_scroll_btn.setFixedWidth(70); self._log_scroll_btn.setCheckable(True)
        self._log_scroll_btn.setChecked(True)
        self._log_scroll_btn.clicked.connect(self._on_log_scroll_toggle)
        self._log_scroll_btn.setStyleSheet(f"""
            QPushButton {{ background: {_th.PANEL}; border: 1px solid {_th.BORDER};
                border-radius: 6px; color: {_th.FG_DIM}; padding: 2px 6px; }}
            QPushButton:checked {{ color: {_th.ACCENT}; border-color: {_th.ACCENT}; }}
            QPushButton:hover {{ background: {_th.SURFACE2}; }}
        """)
        tbh.addWidget(self._log_scroll_btn)

        # Copy button
        self._log_copy_btn = QPushButton("⎘ Copy"); self._log_copy_btn.setFont(FONT_SMALL)
        self._log_copy_btn.setFixedWidth(70)
        self._log_copy_btn.clicked.connect(self._copy_log)
        self._log_copy_btn.setStyleSheet(f"""
            QPushButton {{ background: {_th.PANEL}; border: 1px solid {_th.BORDER};
                border-radius: 6px; color: {_th.FG_DIM}; padding: 2px 6px; }}
            QPushButton:hover {{ background: {_th.SURFACE2}; color: {_th.FG}; }}
        """)
        tbh.addWidget(self._log_copy_btn)

        lv.addWidget(tb)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(FONT_MONO); self.log.setFixedHeight(140)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {_th.MANTLE}; color: {_th.FG};
                border: 1px solid {_th.BORDER}; border-radius: 8px;
                padding: 6px; selection-background-color: {_th.ACCENT};
            }}
        """)
        lv.addWidget(self.log)
        self._root_layout.addWidget(self._log_frame)
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
        self._form_outer = QWidget(); self._form_outer.setStyleSheet(f"background: {_th.BG};")
        outer_h = QHBoxLayout(self._form_outer); outer_h.setContentsMargins(8, 4, 8, 0); outer_h.setSpacing(8)

        # ── LEFT: Setup (single scrollable panel, no tabs) ────────────────────
        self._setup_w = QWidget()
        self._setup_w.setStyleSheet(f"""
            QWidget {{ background: {_th.PANEL}; border-radius: 9px; }}
        """)
        setup_outer = QVBoxLayout(self._setup_w); setup_outer.setContentsMargins(0,0,0,0); setup_outer.setSpacing(0)

        # Setup header label
        self._setup_hdr = QWidget(); self._setup_hdr.setStyleSheet(f"background: {_th.MANTLE}; border-radius: 0px; border-bottom: 1px solid {_th.BORDER};")
        self._setup_hdr.setFixedHeight(38)
        sh = QHBoxLayout(self._setup_hdr); sh.setContentsMargins(16, 0, 16, 0)
        self._lbl_setup = QLabel("Setup"); self._lbl_setup.setFont(FONT_SEC)
        self._lbl_setup.setStyleSheet(f"color: {_th.FG}; background: transparent;")
        sh.addWidget(self._lbl_setup); sh.addStretch()
        setup_outer.addWidget(self._setup_hdr)

        self._sa_setup, inner_setup, self._setup_layout = _make_scroll_widget()
        self._setup_layout.addStretch()
        setup_outer.addWidget(self._sa_setup)

        # Aliases for backward compat (presets, collect_kwargs etc reference these)
        self._input_layout = self._setup_layout
        self._beam_layout  = self._setup_layout

        # ── RIGHT: flat tab bar + stacked pages (Panels, Appearance, Export) ──
        self._right_w = QWidget(); self._right_w.setStyleSheet(f"background: {_th.PANEL};")
        right_v = QVBoxLayout(self._right_w); right_v.setContentsMargins(0, 0, 0, 0); right_v.setSpacing(0)

        # Tab strip — flat underline style matching HTML role="tab" design
        self._tab_strip = QWidget()
        self._tab_strip.setStyleSheet(f"background: {_th.PANEL}; border-bottom: 1px solid {_th.BORDER};")
        tab_strip_h = QHBoxLayout(self._tab_strip)
        tab_strip_h.setContentsMargins(22, 0, 22, 0); tab_strip_h.setSpacing(4)

        self._tab_stack = QStackedWidget()
        self._tab_stack.setStyleSheet(f"background: {_th.PANEL};")

        self._tab_btns = []
        self._tab_pages = []
        for i, name in enumerate(("Panels", "Appearance", "Export")):
            btn = QPushButton(name); btn.setFont(FONT_SEC)
            btn.setCheckable(False)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    border-bottom: 2px solid transparent;
                    color: {_th.FG_DIM}; padding: 14px 18px;
                    font-size: 13px;
                }}
                QPushButton:hover {{ color: {_th.FG}; }}
            """)
            btn.clicked.connect(lambda _=False, idx=i: self._switch_tab(idx))
            tab_strip_h.addWidget(btn)
            self._tab_btns.append(btn)
            page = QWidget(); page.setStyleSheet(f"background: {_th.PANEL};")
            self._tab_stack.addWidget(page)
            self._tab_pages.append(page)

        tab_strip_h.addStretch()
        right_v.addWidget(self._tab_strip)
        right_v.addWidget(self._tab_stack, 1)

        def _scroll_tab_page(idx):
            page = self._tab_stack.widget(idx)
            sa, inner, vbox = _make_scroll_widget()
            vbox.addStretch()
            pl = QVBoxLayout(page); pl.setContentsMargins(0, 0, 0, 0)
            pl.addWidget(sa)
            return sa, vbox

        self._sa_panels,     self._panels_layout     = _scroll_tab_page(0)
        self._sa_appearance, self._appearance_layout = _scroll_tab_page(1)
        self._sa_export,     self._export_layout     = _scroll_tab_page(2)

        # Aliases for backward compat
        self._visual_layout = self._appearance_layout
        self._colors_layout = self._appearance_layout

        # Select first tab
        self._switch_tab(0)

        outer_h.addWidget(self._setup_w, 1)
        outer_h.addWidget(self._right_w, 1)
        self._root_layout.addWidget(self._form_outer, 1)

        # Build content
        self._setup_layout.takeAt(self._setup_layout.count() - 1)
        self._build_input_section(self._setup_layout)
        self._build_beam_section(self._setup_layout)
        self._setup_layout.addStretch(1)

        for lay in (self._panels_layout, self._appearance_layout, self._export_layout):
            lay.takeAt(lay.count() - 1)

        self._build_panels_section(self._panels_layout)
        self._build_visual_section(self._appearance_layout)
        self._build_colors_section(self._appearance_layout)
        self._build_export_section(self._export_layout)

        for lay in (self._panels_layout, self._appearance_layout, self._export_layout):
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
        lbl_u.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent;"); lbl_u.setFixedWidth(160)
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
        _btn(add_row, "+ Add file", self._add_compare_file, width=90, color=_th.ACCENT)

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
                QPushButton {{ background: {_th.BG}; border: 1px solid {_th.BORDER}; border-radius: 6px; color: {_th.FG}; }}
                QPushButton:checked {{ background: {_th.ACCENT}; border-color: {_th.ACCENT}; color: white; }}
                QPushButton:hover   {{ background: {_th.BORDER}; }}
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
        self.w_betagamma.setStyleSheet(f"color: {_th.SUCCESS}; background: transparent;")
        r2.addWidget(self.w_betagamma)

        layout.addWidget(self._norm_widget)
        self._norm_widget.hide()
        self._update_emit_ui()

    # ── Panels section ────────────────────────────────────────────────────────

    def _build_panels_section(self, layout):
        self._panels_layout_ref = layout   # save for overlay swap

        lbl = QLabel("Add panels, click name to rename:"); lbl.setFont(FONT_MAIN)
        lbl.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent; padding: 4px 12px 2px 12px;")
        layout.addWidget(lbl)
        _help(layout, "Stacked plots below the floor plan. ▲▼ to reorder, click name to rename.")

        self._panel_frame_widget = QWidget(); self._panel_frame_widget.setStyleSheet("background: transparent;")
        self._panel_frame_vbox = QVBoxLayout(self._panel_frame_widget)
        self._panel_frame_vbox.setContentsMargins(8, 4, 8, 4); self._panel_frame_vbox.setSpacing(2)
        layout.addWidget(self._panel_frame_widget)
        self._render_panel_list()

        # Add preset section header — uppercase label with line
        _sec(layout, "Add preset")

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
                    background: {_th.PANEL2}; border: 1px solid {_th.BORDER};
                    border-radius: 7px; color: {_th.FG_DIM};
                }}
                QPushButton:hover {{ border-color: {_th.ACCENT}; color: {_th.ACCENT}; }}
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
                    background: transparent;
                    border: 1px dashed {_th.COPPER};
                    border-radius: 7px; color: {_th.COPPER};
                }}
                QPushButton:hover {{ background: {_th.CSOFT}; }}
            """)
            b.clicked.connect(cmd)
            btn_grid.addWidget(b, idx // _NCOLS, idx % _NCOLS)
        layout.addWidget(btn_grid_w)

        _sec(layout, "Panel Options")
        r = _row(layout)
        self.w_grid_enable = _chk(r, "Grid layout")
        self.w_grid_enable.setChecked(self._grid_enabled)
        self.w_grid_enable.toggled.connect(self._on_grid_toggle)
        _lbl(r, "Size", width=35)
        self.w_grid_size = _ent(r, width=55, placeholder="2x2")
        self.w_grid_size.setText(f"{self._grid_rows}x{self._grid_cols}" if hasattr(self, '_grid_rows') else "2x2")
        self.w_grid_size.setToolTip("Grid dimensions, e.g. 2x2, 2x3, 3x3")
        self.w_grid_size.textChanged.connect(self._on_grid_size_changed)
        _hint(r, "rows×cols")
        r = _row(layout)
        self.w_cell_srange = _chk(r, "Custom s-range per panel")
        self.w_cell_srange.setChecked(False)
        self.w_cell_srange.toggled.connect(self._on_cell_srange_toggle)
        _help(layout, "Show per-panel s-range field in each panel row (grid mode). Disables shared x-axis.")
        r = _row(layout)
        self.w_shared_xaxis = _chk(r, "Shared x-axis"); self.w_shared_xaxis.setChecked(True)
        _help(layout, "Link x-axes of all data panels together (pan/zoom synced).")
        r = _row(layout)
        self.w_hide_labels = _chk(r, "Hide axis labels")
        _help(layout, "Suppress axis and panel title labels on the plot.")
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
        r = _row(layout); _lbl(r, 'Legend mode', width=110)
        self.w_legend_mode = QComboBox(); self.w_legend_mode.setFont(FONT_MAIN)
        self.w_legend_mode.setFixedHeight(28); self.w_legend_mode.setFixedWidth(160)
        self.w_legend_mode.setStyleSheet(_th._COMBO_SS)
        for _lm in ['Side (outside)', 'Inside plot', 'Horizontal (bottom)', 'Horizontal (top)']:
            self.w_legend_mode.addItem(_lm)
        r.addWidget(self.w_legend_mode); r.addStretch()

        _sec(layout, "Floor Plan")
        r = _row(layout); _lbl(r, "X-Z elem ratio")
        self.w_elem_h = _ent(r, width=60); self.w_elem_h.setText("0.05")
        _hint(r, "fraction of axis span")
        r = _row(layout); _lbl(r, "Y-Z elem ratio")
        self.w_elem_h_yz = _ent(r, width=60); _hint(r, "blank = same as X-Z")
        r = _row(layout); _lbl(r, "Bar elem ratio")
        self.w_bar_elem_ratio = _ent(r, width=60); self.w_bar_elem_ratio.setText("0.5")
        _hint(r, "fraction of bar y-axis span")
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
            ("Markers in floor",   lambda c: setattr(self, 'w_show_markers',      c)),
            ("Markers in bar",     lambda c: setattr(self, 'w_show_markers_bar',  c)),
            ("Equal aspect (floor)", lambda c: setattr(self, 'w_equal_aspect',    c)),
        ]
        for i, (lbl_txt, setter) in enumerate(_chk_items):
            cb = QCheckBox(lbl_txt); cb.setFont(FONT_MAIN); cb.setStyleSheet(_th._CHK_SS)
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
            l.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent;")
            e = QLineEdit(); e.setFont(FONT_MONO); e.setFixedWidth(42); e.setFixedHeight(24)
            e.setPlaceholderText(placeholder); e.setStyleSheet(_th._ENTRY_SS)
            row_h.addWidget(l); row_h.addWidget(e); row_h.addStretch()
            setattr(self, attr, e)
            _fs_g.addWidget(row_w, r_idx, c_idx)
        layout.addWidget(_fs_w)
        _help(layout, "Leave blank to use Plotly defaults.")

    # ── Export section ────────────────────────────────────────────────────────

    # ── Grid section ─────────────────────────────────────────────────────────

    def _on_cell_srange_toggle(self, checked):
        self._render_panel_list()

    def _on_grid_toggle(self, checked):
        self._grid_enabled = checked
        self._render_panel_list()  # show/hide grid buttons

    def _on_grid_size_changed(self, text):
        try:
            parts = text.lower().split('x')
            self._grid_rows = max(1, int(parts[0]))
            self._grid_cols = max(1, int(parts[1])) if len(parts) > 1 else self._grid_cols
        except (ValueError, IndexError):
            pass
        self._render_panel_list()  # refresh dropdowns with new row/col options

    def _get_grid_layout(self):
        """Build grid_layout dict from per-panel grid assignments."""
        if not self._grid_enabled:
            return None
        cells = []
        for p in self._panels:
            gr = p.get('grid_row')
            gc = p.get('grid_col')
            if gr is not None and gc is not None:
                spec = p.get('spec', 'twiss')
                # Same height-lookup key as _render_panel_list(): spec string
                # for presets, else the panel's _id/type (spec itself may be an
                # unhashable dict for custom/expression panels).
                h_key = spec if isinstance(spec, str) else spec.get('_id', spec.get('type', 'custom'))
                cells.append({
                    'row': gr,
                    'col': gc,
                    'spec': spec,
                    'name': p.get('name', ''),
                    'source': p.get('grid_source', None),
                    'span_cols': p.get('grid_span', 1),
                    'hide_legend': p.get('hide_legend', False),
                    'annot_pattern': p.get('annot_pattern', ''),
                    'panel_source': p.get('panel_source', None),
                    'cell_srange':  p.get('cell_srange', ''),
                    'height_px':    int(self._panel_height_edits.get(h_key, 300) or 300),
                })
        if not cells:
            return None
        # Per-row height: tallest panel in each row
        _row_heights = {}
        for _c in cells:
            _rr = _c['row']
            _h = _c.get('height_px', 300)
            _row_heights[_rr] = max(_row_heights.get(_rr, 0), _h)
        return {
            'rows': self._grid_rows,
            'cols': self._grid_cols,
            'row_heights': [_row_heights.get(r+1, 300) for r in range(self._grid_rows)],
            'cells': cells,
        }

    # ── Colors section ───────────────────────────────────────────────────────

    def _build_colors_section(self, layout):
        from core.utils import _ELEM_COLOR_DEFAULTS, _ELEM_COLOR_FALLBACK
        _sec(layout, "Element Colors")
        _help(layout, "Click a swatch to change the color for that element type. "
                      "Colors are saved with presets.")

        _ELEM_LABELS = [
            ('sbend',      'Dipole'),
            ('quadrupole', 'Quadrupole'),
            ('sextupole',  'Sextupole'),
            ('kicker',     'Kicker'),
            ('monitor',    'Monitor'),
            ('marker',     'Marker'),
            ('rfcavity',   'RF Cavity'),
            ('lcavity',    'Linac Cavity'),
        ]

        for key, label in _ELEM_LABELS:
            r = _row(layout)
            _lbl(r, label, width=120)
            default = _ELEM_COLOR_DEFAULTS.get(key, _ELEM_COLOR_FALLBACK)
            current = self._elem_colors.get(key, default)
            btn = QPushButton()
            btn.setFixedSize(48, 24)
            btn.setToolTip(f"Click to change {label} color")
            self._set_swatch(btn, current)
            btn.clicked.connect(lambda checked, k=key, b=btn: self._pick_color(k, b))
            self._color_btns[key] = btn
            r.addWidget(btn)

            # Reset button
            rst = QPushButton("↺")
            rst.setFixedSize(28, 24)
            rst.setFont(FONT_MAIN)
            rst.setToolTip(f"Reset to default")
            rst.setStyleSheet(f"""
                QPushButton {{
                    background: {_th.PANEL}; border: 1px solid {_th.BORDER};
                    border-radius: 6px; color: {_th.FG_DIM};
                }}
                QPushButton:hover {{ color: {_th.ACCENT}; border-color: {_th.ACCENT}; }}
            """)
            rst.clicked.connect(lambda checked, k=key, b=btn, d=default: self._reset_color(k, b, d))
            r.addWidget(rst)
            r.addStretch()

    def _set_swatch(self, btn, color):
        """Set swatch button background to color."""
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color}; border: 1px solid {_th.BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{ border-color: {_th.ACCENT}; }}
        """)
        btn.setProperty("color", color)

    def _pick_color(self, key, btn):
        """Open color dialog and update swatch + _elem_colors."""
        from core.utils import _ELEM_COLOR_DEFAULTS, _ELEM_COLOR_FALLBACK
        current = self._elem_colors.get(key, _ELEM_COLOR_DEFAULTS.get(key, _ELEM_COLOR_FALLBACK))
        from PySide6.QtGui import QColor
        init = QColor(current)
        chosen = QColorDialog.getColor(init, self, f"Choose color")
        if chosen.isValid():
            hex_color = chosen.name()
            self._elem_colors[key] = hex_color
            self._set_swatch(btn, hex_color)

    def _reset_color(self, key, btn, default):
        """Reset element color to default."""
        self._elem_colors.pop(key, None)
        self._set_swatch(btn, default)

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
        self._panel_grid_checks = {}
        n = len(self._panels)
        _DEFAULT_H = {'floor-xz': 220, 'floor-yz': 220, 'bar': 80,
                      'latdiff': 260, 'summary': 260}
        for pos, panel in enumerate(self._panels):
            # Panel row — styled as per mockup: bg panel2, border, rounded
            row_w = QWidget()
            row_w.setStyleSheet(f"""
                QWidget {{ background: {_th.PANEL2}; border: 1px solid {_th.BORDER};
                    border-radius: 9px; }}
            """)
            rh = QHBoxLayout(row_w); rh.setContentsMargins(11, 7, 11, 7); rh.setSpacing(8)

            # Drag handle
            drag_lbl = QLabel('⠿'); drag_lbl.setFont(FONT_MAIN)
            drag_lbl.setFixedWidth(14)
            drag_lbl.setStyleSheet(f'color: {_th.FG_DIM}; background: transparent; border: none;')
            rh.addWidget(drag_lbl)

            # Panel name — click to rename
            name_btn = QPushButton(panel['name']); name_btn.setFont(FONT_MAIN)
            name_btn.setMinimumWidth(120); name_btn.setFixedHeight(26)
            name_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; color: {_th.FG};
                    text-align: left; padding: 0 4px; font-weight: 500; }}
                QPushButton:hover {{ color: {_th.ACCENT}; }}
            """)
            name_btn.clicked.connect(lambda _=False, p=pos: self._rename_panel(p))
            rh.addWidget(name_btn, 1)

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
            # Height field — styled container with inline px label
            h_box = QWidget()
            h_box.setStyleSheet(f"background: {_th.MANTLE}; border: 1px solid {_th.BORDER}; border-radius: 6px;")
            h_bl = QHBoxLayout(h_box); h_bl.setContentsMargins(8, 3, 8, 3); h_bl.setSpacing(4)
            h_edit = QLineEdit(prev_val); h_edit.setFixedWidth(32); h_edit.setFixedHeight(22)
            h_edit.setFont(FONT_MONO)
            h_edit.setStyleSheet(f"border: none; background: transparent; color: {_th.FG}; padding: 0;")
            h_edit.setToolTip("Panel height in pixels")
            self._panel_height_edits[h_key] = prev_val
            h_edit.textChanged.connect(lambda v, k=h_key: self._panel_height_edits.update({k: v}))
            px_lbl = QLabel('px'); px_lbl.setFont(FONT_SMALL)
            px_lbl.setStyleSheet(f'color: {_th.FG_DIM}; background: transparent; border: none;')
            h_bl.addWidget(h_edit); h_bl.addWidget(px_lbl)
            rh.addWidget(h_box)

            for sym, cmd, col in [
                ("▲", lambda _=False, p=pos: self._move_panel(p, -1), _th.FG_DIM),
                ("▼", lambda _=False, p=pos: self._move_panel(p, +1), _th.FG_DIM),
                ("✕", lambda _=False, p=pos: self._remove_panel(p),   _th.ERROR),
            ]:
                disabled = (sym == "▲" and pos == 0) or (sym == "▼" and pos == n - 1)
                b = QPushButton(sym); b.setFixedSize(28, 28); b.setFont(FONT_SMALL)
                b.setEnabled(not disabled)
                b.setStyleSheet(f"""
                    QPushButton {{ background: transparent; border: 1px solid {_th.BORDER};
                        border-radius: 6px; color: {_th.FG_DIM}; }}
                    QPushButton:hover:enabled {{ color: {col}; border-color: {col}; }}
                    QPushButton:disabled {{ color: {_th.BORDER}; border-color: transparent; }}
                """)
                b.clicked.connect(cmd)
                rh.addWidget(b)
            # ⋯ overflow button — reveals extra controls
            # Universe selector — always visible in main row
            _uni_dd = QComboBox(); _uni_dd.setFont(FONT_SMALL)
            _uni_dd.setFixedWidth(60); _uni_dd.setFixedHeight(26)
            _uni_dd.setStyleSheet(_th._COMBO_SS)
            _uni_dd.setToolTip('Universe for this panel (All = global selection)')
            _uni_dd.blockSignals(True)  # prevent spurious panel_source overwrite during init
            _uni_dd.addItem('All', None)
            if hasattr(self, '_uni_checks') and self._uni_checks:
                for _uid in sorted(self._uni_checks.keys()):
                    try:
                        _ulbl = self._uni_label_edits[_uid].text().strip() if hasattr(self, '_uni_label_edits') and _uid in self._uni_label_edits else f'u{_uid}'
                    except RuntimeError:
                        _ulbl = f'u{_uid}'  # widget deleted during theme switch
                    _uni_dd.addItem(f'u{_uid}', _uid)
            _cur_psrc = panel.get('panel_source', None)
            if _cur_psrc is not None:
                for _idx in range(_uni_dd.count()):
                    if _uni_dd.itemData(_idx) == _cur_psrc:
                        _uni_dd.setCurrentIndex(_idx); break
            _uni_dd.blockSignals(False)  # re-enable signals after init
            _uni_dd.currentIndexChanged.connect(
                lambda idx, p=pos, dd=_uni_dd: self._panels[p].update({'panel_source': dd.currentData()}))
            rh.addWidget(_uni_dd)

            _more_btn = QPushButton('⋯'); _more_btn.setFixedSize(26, 26)
            _more_btn.setFont(FONT_SMALL); _more_btn.setCheckable(True)
            _more_btn.setChecked(False)
            _more_btn.setToolTip('More options')
            _more_btn.setStyleSheet(f"""
                QPushButton {{ background: {_th.BORDER}; border-radius: 4px; color: {_th.FG_DIM}; border: none; }}
                QPushButton:checked {{ background: {_th.ASOFT}; color: {_th.ACCENT}; }}
                QPushButton:hover {{ background: {_th.SURFACE2}; color: {_th.FG}; }}
            """)
            rh.addWidget(_more_btn)

            # Overflow container — hidden by default
            _ovf = QWidget(); _ovf.setStyleSheet('background: transparent;')
            _ovf_h = QHBoxLayout(_ovf); _ovf_h.setContentsMargins(4, 0, 0, 0); _ovf_h.setSpacing(4)
            _ovf.setVisible(False)
            _more_btn.toggled.connect(_ovf.setVisible)
            rh.addWidget(_ovf)

            # From here, all extra widgets go into _ovf_h instead of rh
            rh = _ovf_h

            # Y-Z ring half selector — only for floor-yz panels
            if spec == 'floor-yz':
                _yz_half_val = panel.get('yz_ring_half', 'full')
                _yz_half_opts = {'full': '½ Full', 'first': '+X', 'second': '-X'}
                yz_half_btn = QPushButton(_yz_half_opts[_yz_half_val])
                yz_half_btn.setFixedHeight(26); yz_half_btn.setFont(FONT_SMALL)
                yz_half_btn.setToolTip('Ring half: Full / +X side (X≥0) / -X side (X<0)')
                yz_half_btn.setStyleSheet(f"""
                    QPushButton {{ background: {_th.BORDER}; border-radius: 4px;
                        color: {_th.FG}; border: none; padding: 0 4px; }}
                    QPushButton:hover {{ background: {_th.ACCENT2}; color: white; }}
                """)
                def _cycle_yz_half(checked=False, p=pos, btn=yz_half_btn):
                    _cycle = {'full': 'first', 'first': 'second', 'second': 'full'}
                    _labels = {'full': '½ Full', 'first': '+X', 'second': '-X'}
                    cur = self._panels[p].get('yz_ring_half', 'full')
                    nxt = _cycle[cur]
                    self._panels[p]['yz_ring_half'] = nxt
                    btn.setText(_labels[nxt])
                yz_half_btn.clicked.connect(_cycle_yz_half)
                rh.addWidget(yz_half_btn)

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
                    QPushButton {{ background: {_th.BORDER}; border-radius: 4px; color: {_th.FG}; border: none; }}
                    QPushButton:checked {{ background: {_th.ACCENT2}; color: white; }}
                    QPushButton:hover {{ background: {_th.ACCENT2}; color: white; }}
                """)
                rh.addWidget(annot_btn)

                # Pattern field — visible only when toggled on
                annot_edit = QLineEdit(panel.get('annot_pattern', ''))
                annot_edit.setFixedWidth(90); annot_edit.setFixedHeight(22)
                annot_edit.setFont(FONT_MAIN); annot_edit.setStyleSheet(_th._ENTRY_SS)
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
                    QPushButton {{ background: {_th.BORDER}; border-radius: 4px; color: {_th.FG}; border: none; }}
                    QPushButton:checked {{ background: {_th.HIGHLIGHT}; color: {_th.CRUST}; }}
                    QPushButton:hover {{ background: {_th.HIGHLIGHT}; color: {_th.CRUST}; }}
                """)
                rh.addWidget(leg_btn)

                leg_x_edit = QLineEdit(); leg_x_edit.setFixedWidth(36); leg_x_edit.setFixedHeight(22)
                leg_x_edit.setFont(FONT_MAIN); leg_x_edit.setStyleSheet(_th._ENTRY_SS)
                leg_x_edit.setPlaceholderText('X')
                leg_x_edit.setText(str(panel.get('legend_x', '')))
                leg_x_edit.setVisible(False)
                rh.addWidget(leg_x_edit)

                leg_y_edit = QLineEdit(); leg_y_edit.setFixedWidth(36); leg_y_edit.setFixedHeight(22)
                leg_y_edit.setFont(FONT_MAIN); leg_y_edit.setStyleSheet(_th._ENTRY_SS)
                leg_y_edit.setPlaceholderText('Y')
                leg_y_edit.setText(str(panel.get('legend_y', '')))
                leg_y_edit.setVisible(False)
                rh.addWidget(leg_y_edit)

                def _on_leg_toggle(checked, xe=leg_x_edit, ye=leg_y_edit, p=pos):
                    xe.setVisible(checked)
                    ye.setVisible(checked)
                    if checked:
                        self._panels[p]['legend_x'] = xe.text().strip()
                        self._panels[p]['legend_y'] = ye.text().strip()
                    else:
                        self._panels[p].pop('legend_x', None)
                        self._panels[p].pop('legend_y', None)
                def _on_leg_x_changed(text, p=pos):
                    self._panels[p]['legend_x'] = text.strip()
                def _on_leg_y_changed(text, p=pos):
                    self._panels[p]['legend_y'] = text.strip()
                leg_x_edit.textChanged.connect(_on_leg_x_changed)
                leg_y_edit.textChanged.connect(_on_leg_y_changed)
                # Restore saved state if available
                if _leg_key in _saved_leg:
                    _sx, _sy = _saved_leg[_leg_key]
                    leg_btn.setChecked(True)
                    leg_x_edit.setText(_sx)
                    leg_y_edit.setText(_sy)
                    leg_x_edit.setVisible(True)
                    leg_y_edit.setVisible(True)
                elif panel.get('legend_x', '') or panel.get('legend_y', ''):
                    leg_btn.setChecked(True)
                    leg_x_edit.setVisible(True)
                    leg_y_edit.setVisible(True)

                leg_btn.toggled.connect(_on_leg_toggle)

                self._panel_leg_checks[_leg_key]  = leg_btn
                self._panel_leg_x_edits[_leg_key] = leg_x_edit
                self._panel_leg_y_edits[_leg_key] = leg_y_edit

            # Per-cell s-range field — visible only when custom s-range is enabled
            if hasattr(self, 'w_cell_srange') and self.w_cell_srange.isChecked():
                sr_lbl = QLabel('s:'); sr_lbl.setFont(FONT_SMALL)
                sr_lbl.setStyleSheet(f'color: {_th.FG_DIM}; background: transparent; border: none;')
                rh.addWidget(sr_lbl)
                sr_edit = QLineEdit(); sr_edit.setFixedWidth(110); sr_edit.setFixedHeight(24)
                sr_edit.setPlaceholderText('START:END'); sr_edit.setFont(FONT_MONO)
                sr_edit.setStyleSheet(_th._ENTRY_SS)
                sr_edit.setText(panel.get('cell_srange', ''))
                sr_edit.setToolTip('s-range for this panel: START:END (m or element name)')
                sr_edit.textChanged.connect(
                    lambda v, p=pos: self._panels[p].update({'cell_srange': v.strip()}))
                rh.addWidget(sr_edit)

            # Hide legend button
            hide_leg_btn = QPushButton('∅'); hide_leg_btn.setFixedSize(26, 26)
            hide_leg_btn.setFont(FONT_SMALL); hide_leg_btn.setCheckable(True)
            hide_leg_btn.setChecked(panel.get('hide_legend', False))
            hide_leg_btn.setToolTip('Hide legend for this panel')
            hide_leg_btn.setStyleSheet(f"""
                QPushButton {{ background: {_th.BORDER}; border-radius: 4px; color: {_th.FG}; border: none; }}
                QPushButton:checked {{ background: {_th.ERROR}; color: white; }}
                QPushButton:hover {{ background: {_th.SURFACE2}; }}
            """)
            hide_leg_btn.toggled.connect(lambda checked, p=pos: self._panels[p].update({'hide_legend': checked}))
            rh.addWidget(hide_leg_btn)

            # Grid placement button — visible only when grid is enabled
            if self._grid_enabled:
                grid_btn = QPushButton('⊞'); grid_btn.setFixedSize(26, 26)
                grid_btn.setFont(FONT_SMALL); grid_btn.setCheckable(True)
                _has_grid = panel.get('grid_row') is not None
                grid_btn.setChecked(_has_grid)
                grid_btn.setToolTip('Set grid position (row, col) and universe')
                grid_btn.setStyleSheet(f"""
                    QPushButton {{ background: {_th.BORDER}; border-radius: 4px; color: {_th.FG}; border: none; }}
                    QPushButton:checked {{ background: {_th.ACCENT}; color: {_th.CRUST}; }}
                    QPushButton:hover {{ background: {_th.ACCENT}; color: {_th.CRUST}; }}
                """)
                rh.addWidget(grid_btn)

                # Row dropdown
                grid_row_dd = QComboBox(); grid_row_dd.setFont(FONT_SMALL)
                grid_row_dd.setFixedWidth(38); grid_row_dd.setFixedHeight(24)
                grid_row_dd.setStyleSheet(_th._COMBO_SS + "QComboBox { padding: 2px 4px; }")
                grid_row_dd.setToolTip('Grid row')
                for ri in range(1, self._grid_rows + 1):
                    grid_row_dd.addItem(str(ri), ri)
                _cur_gr = panel.get('grid_row', 1)
                grid_row_dd.setCurrentIndex(max(0, _cur_gr - 1))
                grid_row_dd.setVisible(_has_grid)
                rh.addWidget(grid_row_dd)

                # Col dropdown
                grid_col_dd = QComboBox(); grid_col_dd.setFont(FONT_SMALL)
                grid_col_dd.setFixedWidth(38); grid_col_dd.setFixedHeight(24)
                grid_col_dd.setStyleSheet(_th._COMBO_SS + "QComboBox { padding: 2px 4px; }")
                grid_col_dd.setToolTip('Grid column')
                for ci in range(1, self._grid_cols + 1):
                    grid_col_dd.addItem(str(ci), ci)
                _cur_gc = panel.get('grid_col', 1)
                grid_col_dd.setCurrentIndex(max(0, _cur_gc - 1))
                grid_col_dd.setVisible(_has_grid)
                rh.addWidget(grid_col_dd)

                # Universe dropdown
                grid_uni_dd = QComboBox(); grid_uni_dd.setFont(FONT_SMALL)
                grid_uni_dd.setFixedWidth(70); grid_uni_dd.setFixedHeight(22)
                grid_uni_dd.setStyleSheet(_th._COMBO_SS + "QComboBox { padding: 2px 4px; }")
                grid_uni_dd.setToolTip('Universe (All = selected universes)')
                grid_uni_dd.blockSignals(True)
                grid_uni_dd.addItem('All', None)
                if hasattr(self, '_uni_checks') and self._uni_checks:
                    for _uid in sorted(self._uni_checks.keys()):
                        _ulbl = self._uni_label_edits[_uid].text().strip() if hasattr(self, '_uni_label_edits') and _uid in self._uni_label_edits else f'u{_uid}'
                        grid_uni_dd.addItem(f'u{_uid}:{_ulbl}', _uid)
                _cur_src = panel.get('grid_source', None)
                if _cur_src is not None:
                    for _i in range(grid_uni_dd.count()):
                        if grid_uni_dd.itemData(_i) == _cur_src:
                            grid_uni_dd.setCurrentIndex(_i); break
                grid_uni_dd.blockSignals(False)
                grid_uni_dd.setVisible(_has_grid)
                rh.addWidget(grid_uni_dd)

                def _on_grid_btn_toggle(checked, p=pos,
                                        rdd=grid_row_dd, cdd=grid_col_dd, udd=grid_uni_dd):
                    rdd.setVisible(checked)
                    cdd.setVisible(checked)
                    udd.setVisible(checked)
                    if checked:
                        # If this panel has no grid position yet, auto-assign the next
                        # free slot in row-major order (1,1 -> 1,2 -> ... -> 2,1 -> ...).
                        # Manual override via the dropdowns still works after assignment.
                        if self._panels[p].get('grid_row') is None:
                            _taken = {(pp.get('grid_row'), pp.get('grid_col'))
                                      for ppi, pp in enumerate(self._panels)
                                      if ppi != p and pp.get('grid_row') is not None}
                            _next = None
                            for _rr in range(1, self._grid_rows + 1):
                                for _cc in range(1, self._grid_cols + 1):
                                    if (_rr, _cc) not in _taken:
                                        _next = (_rr, _cc); break
                                if _next: break
                            if _next is None:
                                _next = (1, 1)  # grid is full — fall back to (1,1)
                            rdd.setCurrentIndex(max(0, _next[0] - 1))
                            cdd.setCurrentIndex(max(0, _next[1] - 1))
                        self._panels[p]['grid_row'] = rdd.currentData()
                        self._panels[p]['grid_col'] = cdd.currentData()
                        self._panels[p]['grid_source'] = udd.currentData()
                    else:
                        self._panels[p].pop('grid_row', None)
                        self._panels[p].pop('grid_col', None)
                        self._panels[p].pop('grid_source', None)

                def _on_grid_row_change(idx, p=pos, rdd=grid_row_dd):
                    self._panels[p]['grid_row'] = rdd.currentData()
                def _on_grid_col_change(idx, p=pos, cdd=grid_col_dd):
                    self._panels[p]['grid_col'] = cdd.currentData()
                def _on_grid_uni_change(idx, p=pos, udd=grid_uni_dd):
                    self._panels[p]['grid_source'] = udd.currentData()

                grid_btn.toggled.connect(_on_grid_btn_toggle)
                grid_row_dd.currentIndexChanged.connect(_on_grid_row_change)
                grid_col_dd.currentIndexChanged.connect(_on_grid_col_change)
                grid_uni_dd.currentIndexChanged.connect(_on_grid_uni_change)
                self._panel_grid_checks[pos] = grid_btn

            # Add stretch to overflow container, not main row
            rh.addStretch()
            # The main row_w layout already has the stretch via _ovf
            row_w.layout().addStretch(1)
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
        if ok:
            self._panels[pos]['name'] = name.strip()
            self._render_panel_list()

    def _get_panels(self):
        return [p['spec'] for p in self._panels] if self._panels else ['twiss']

    def _get_panel_annotations(self):
        """Return {panel_index: pattern} matching the engine's reordered panel list."""
        # Engine reorders: floor panels first, bar last — match that order
        # Use a safe string key for each panel (spec string for presets, _id for dicts)
        def _spec_key(p):
            s = p['spec']
            if isinstance(s, str): return s
            return s.get('_id', id(s))  # use _id or object id for dict specs

        panels = self._panels or []
        def _is_floor(p): return isinstance(p['spec'], str) and p['spec'] in ('floor-xz', 'floor-yz')
        def _is_bar(p):   return p['spec'] == 'bar'
        _floor = [p for p in panels if _is_floor(p)]
        _bar   = [p for p in panels if _is_bar(p)]
        _data  = [p for p in panels if not _is_floor(p) and not _is_bar(p)]
        reordered = _floor + _data + _bar

        result = {}
        for i, p in enumerate(reordered):
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
        tab_page = self._tab_stack.widget(0)   # "Panels" page
        # Hide existing layout widget
        old_sa = tab_page.layout().itemAt(0).widget()
        old_sa.hide()

        overlay_w = QWidget(); overlay_w.setStyleSheet(f"background: {_th.BG};")
        overlay_v = QVBoxLayout(overlay_w); overlay_v.setContentsMargins(0, 0, 0, 0)
        tab_page.layout().addWidget(overlay_w)

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
            lbl_e.setFixedWidth(100); lbl_e.setStyleSheet(_th._ENTRY_SS)
            lbl_e.setToolTip("Display label")
            lbl_e.textChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'label': t}))
            row_h.addWidget(lbl_e)

            # File path (truncated display)
            path_lbl = QLabel(Path(entry['file']).name); path_lbl.setFont(FONT_SMALL)
            path_lbl.setStyleSheet(f"color: {_th.FG_DIM}; background: transparent;")
            path_lbl.setToolTip(entry['file'])
            row_h.addWidget(path_lbl)

            # Code badge
            code_dd = QComboBox(); code_dd.setFont(FONT_SMALL)
            code_dd.addItems(["tao", "elegant", "xsuite"])
            code_dd.setCurrentText(entry['code']); code_dd.setFixedWidth(82)
            code_dd.setStyleSheet(_th._COMBO_SS)
            code_dd.currentTextChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'code': t}))
            row_h.addWidget(code_dd)

            # Remove button
            rm = QPushButton("✕"); rm.setFixedSize(22, 22); rm.setFont(FONT_SMALL)
            rm.setStyleSheet(f"QPushButton {{ background: {_th.BORDER}; color: {_th.FG_DIM}; border: none; border-radius: 3px; }}"
                             f"QPushButton:hover {{ background: {_th.ERROR}; color: white; }}")
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
                ul.setStyleSheet(f"color: {_th.FG_LBL}; background: transparent;")
                uni_row_h.addWidget(ul)
                uni_checks = {}
                labels = entry.get('uni_labels', {})
                for ui in range(1, n + 1):
                    cb = QCheckBox(f"u{ui}:{labels.get(ui, f'u{ui}')}")
                    cb.setChecked(True); cb.setFont(FONT_SMALL); cb.setStyleSheet(_th._CHK_SS)
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
            cb.setStyleSheet(_th._CHK_SS); self._uni_checks[i] = cb
            cell_h.addWidget(cb)
            le = QLineEdit(lbl); le.setFixedWidth(100); le.setFont(FONT_MAIN)
            le.setStyleSheet(_th._ENTRY_SS)
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
            bar_elem_ratio=_fn(self.w_bar_elem_ratio) or 0.5,
            fp_xz_range=fp_xz, fp_yz_range=fp_yz,
            panels=self._get_panels(), layout="panels", srange=rng,
            panel_annotations=self._get_panel_annotations(),
            legend_positions=self._get_legend_positions(),
            font_sizes=self._get_font_sizes(),
            panel_heights=self._get_panel_heights(),
            emit_x=ex, emit_y=ey, sigma_dp=_fn(self.w_sigmadp),
            n_sigma=_fn(self.w_nsigma) or 1.0, title=ttl,
            dark_mode=self.w_dark.isChecked(), aspect_ratio=self.w_aspect.text().strip() or None,
            legend_inside=self.w_legend_mode.currentIndex() == 1,
            legend_mode=['side','inside','bottom','top'][self.w_legend_mode.currentIndex()],
            panels_meta={i: {'hide_legend': p.get('hide_legend', False), 'panel_source': p.get('panel_source', None)} for i, p in enumerate(self._panels)},
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
            equal_aspect=self.w_equal_aspect.isChecked(),
            yz_ring_half=next((p.get('yz_ring_half','full') for p in self._panels if p.get('spec')=='floor-yz'), 'full'),
            panel_metadata=self._panels,
            show_titles=self.w_show_titles.isChecked(),
            panel_spacing=float(self.w_panel_spacing.text().strip() or '80'),
            compare=self._get_compare_list(),
            compare_mode=self.w_compare_mode.currentText().lower().replace(' ', '').replace('(%)', '%'),
            normalize_s=self.w_normalize_s.isChecked(),
            elem_colors=self._elem_colors if self._elem_colors else None,
            grid_layout=self._get_grid_layout(),
            shared_xaxis=self.w_shared_xaxis.isChecked(),
            hide_labels=self.w_hide_labels.isChecked(),
        )

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return
        _missing = check_backend_ready(kwargs['code'])
        if _missing:
            QMessageBox.critical(self, "Backend Not Found", _missing); return

        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self.open_btn.setStyleSheet("")  # reset green highlight from previous run
        self._set_status("Running…"); self._progress.setValue(0); self._pct_lbl.setText("0%")
        self._log("\n" + "─" * 60 + "\n", "dim")
        self._log(f"▶ code={kwargs['code']}  panels={kwargs['panels']}\n", "info")
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
                background: {_th.SUCCESS}; border: 1px solid {_th.SUCCESS};
                border-radius: 8px; color: {_th.CRUST}; font-weight: bold;
            }}
            QPushButton:hover {{ background: {_th.SUCCESS}; color: {_th.CRUST}; opacity: 0.9; }}
        """)
        self._save_recent(input_file)

    def _on_run_failed(self, _unused):
        pass   # status already set via _sig_progress

    def _on_run_finally(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.dryrun_btn.setEnabled(True)
        self.csv_btn.setEnabled(True)
        self._progress.setValue(0)
        self._pct_lbl.setText("")

    def _progress_safe(self, pct, label=None):
        """Thread-safe: emit signal, Qt delivers it to main thread."""
        self._sig_progress.emit(int(pct), label or "")

    def _dry_run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return
        _missing = check_backend_ready(kwargs['code'])
        if _missing:
            QMessageBox.critical(self, "Backend Not Found", _missing); return

        self.run_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self._set_status("Inspecting…")
        self._log("\n🔍 Dry run — loading lattice only…\n", "info")

        def _worker():
            try:
                code = kwargs['code']; inp = kwargs['input_file']
                log = lambda m: self._sig_log.emit(m, "info")
                if code == 'tao':      data = load_tao(inp, log_fn=log)
                elif code == 'xsuite': data = load_xsuite(inp, log_fn=log)
                elif code == 'madx':   data = load_madx(inp, survey_file=kwargs.get('madx_survey'), log_fn=log)
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
        _missing = check_backend_ready(kwargs['code'])
        if _missing:
            QMessageBox.critical(self, "Backend Not Found", _missing); return
        # Use the base name from the field, trigger via save_csv flag
        kwargs['save_csv'] = True
        kwargs['csv_base'] = self.w_csv_base.text().strip() or 'lattice'
        self.csv_btn.setEnabled(False)
        self._set_status("Exporting CSV…")
        def _worker():
            try:
                plot_optics(**kwargs)
                self._sig_progress.emit(0, "CSV exported ✓")
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ CSV error:\n{tb}\n", "error")
                self._sig_progress.emit(0, "Failed")
            finally:
                self._sig_finally.emit()
        threading.Thread(target=_worker, daemon=True).start()

    def _copy_path(self):
        if self._last_output:
            QApplication.clipboard().setText(self._last_output)
            self._set_status("Path copied ✓")
        else:
            QMessageBox.warning(self, "Copy Path", "No output yet. Run first.")

    # ── Log ───────────────────────────────────────────────────────────────────

    @property
    def _LOG_COLORS(self):
        return {"ok": _th.SUCCESS, "warn": _th.WARN, "error": _th.ERROR, "dim": _th.FG_DIM, "info": _th.FG}

    def _write_log(self, text, tag):
        """Write text directly to the log widget with color."""
        color = self._LOG_COLORS.get(tag, _th.FG)
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
            color = self._LOG_COLORS.get(tag, _th.FG)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(_th.FG_DIM))
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
            'range':      self.w_range.text(),
            'panels':     [dict(p) for p in self._panels],
            'emit_type':  'normalized' if self.w_emit_norm.isChecked() else 'geometric',
            'emitx':      self.w_emitx.text(),     'emity':    self.w_emity.text(),
            'sigmadp':    self.w_sigmadp.text(),   'nsigma':   self.w_nsigma.text(),
            'particle':   self.w_particle.currentText(), 'energy': self.w_energy.text(),
            'title':      self.w_title.text(),     'elem_h':   self.w_elem_h.text(),
            'elem_h_yz':  self.w_elem_h_yz.text(), 'fp_xz_range': self.w_fp_xz_range.text(),
            'bar_elem_ratio': self.w_bar_elem_ratio.text(),
            'fp_yz_range': self.w_fp_yz_range.text(),

            'no_labels':  self.w_no_labels.isChecked(), 'flip_bend': self.w_flip_bend.isChecked(),
            'dark_mode':  self.w_dark.isChecked(), 'png': self.w_png.isChecked(),
            'pdf':        self.w_pdf.isChecked(),  'dpi': self.w_dpi.text(),
            'csv_base': self.w_csv_base.text().strip(),
            'aspect':     self.w_aspect.text(),    'legend_mode': self.w_legend_mode.currentIndex(),
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
            'equal_aspect':    self.w_equal_aspect.isChecked(),
            'show_titles':     self.w_show_titles.isChecked(),
            'panel_spacing':   self.w_panel_spacing.text().strip(),
            'madx_survey':     self.w_madx_survey.text().strip(),
            'elem_colors':     dict(self._elem_colors),
            'shared_xaxis':    self.w_shared_xaxis.isChecked(),
            'grid_enabled':    self._grid_enabled,
            'cell_srange_on':  self.w_cell_srange.isChecked(),
            'grid_size':       self.w_grid_size.text().strip(),
        }

    def _preset_defaults(self):
        """Baseline values matching a freshly-constructed GUI (see __init__
        and the _build_*_section methods). Applied before a loaded preset so
        that fields the preset file doesn't mention reset to their default
        instead of silently keeping whatever was on screen before the load."""
        return {
            'code': 'tao', 'output': 'optics.html', 'range': '',
            'panels': [{'name': 'Floor Plan X-Z',      'spec': 'floor-xz'},
                       {'name': 'Twiss & Dispersion', 'spec': 'twiss'},
                       {'name': 'Beamline Bar',       'spec': 'bar'}],
            'emit_type': 'geometric',
            'emitx': '', 'emity': '', 'sigmadp': '', 'nsigma': '1',
            'particle': 'Electron', 'energy': '',
            'title': '', 'elem_h': '0.05', 'elem_h_yz': '', 'fp_xz_range': '',
            'bar_elem_ratio': '0.5', 'fp_yz_range': '',
            'no_labels': False, 'flip_bend': False,
            'dark_mode': False, 'png': False, 'pdf': False, 'dpi': '300',
            'csv_base': '', 'aspect': '', 'legend_mode': 0,
            'compare_files': [], 'compare_mode': 'Overlay',
            'normalize_s': False, 'color_beampipes': False,
            'show_markers': False, 'show_markers_bar': False, 'bar_lite': False,
            'fs_axis': '', 'fs_tick': '', 'fs_title': '', 'fs_annot': '', 'fs_legend': '',
            'show_xz': True, 'show_yz': True, 'equal_aspect': False,
            'show_titles': True, 'panel_spacing': '80', 'madx_survey': '',
            'elem_colors': {}, 'shared_xaxis': True,
            'grid_enabled': False, 'cell_srange_on': False, 'grid_size': '2x2',
        }

    def _apply_preset(self, data):
        data = {**self._preset_defaults(), **data}
        def _st(widget, key):
            if key in data and hasattr(widget, 'setText'): widget.setText(str(data[key]))
        def _sc(widget, key):
            if key in data and hasattr(widget, 'setChecked'): widget.setChecked(bool(data[key]))
        def _sct(widget, key):
            if key in data and hasattr(widget, 'setCurrentText'): widget.setCurrentText(str(data[key]))

        _sct(self.w_code,     'code');     _st(self.w_output, 'output')
        _st(self.w_range,  'range')
        _st(self.w_emitx,  'emitx');       _st(self.w_emity,  'emity')
        _st(self.w_sigmadp,'sigmadp');     _st(self.w_nsigma, 'nsigma')
        _sct(self.w_particle,'particle');  _st(self.w_energy, 'energy')
        _st(self.w_title,  'title');       _st(self.w_elem_h, 'elem_h')
        _st(self.w_elem_h_yz,'elem_h_yz'); _st(self.w_fp_xz_range,'fp_xz_range')
        _st(self.w_bar_elem_ratio,'bar_elem_ratio')
        _st(self.w_fp_yz_range,'fp_yz_range')
        _sc(self.w_no_labels,'no_labels'); _sc(self.w_flip_bend,'flip_bend')
        _sc(self.w_dark,'dark_mode');      _sc(self.w_png,'png');  _sc(self.w_pdf,'pdf')
        _st(self.w_dpi,'dpi');             _st(self.w_aspect,'aspect')
        _st(self.w_csv_base, 'csv_base')
        if 'legend_mode' in data:
            self.w_legend_mode.setCurrentIndex(int(data['legend_mode']))
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
        _sc(self.w_equal_aspect, 'equal_aspect')

        _sc(self.w_show_titles, 'show_titles')
        _st(self.w_panel_spacing, 'panel_spacing')
        if 'madx_survey' in data: self.w_madx_survey.setText(str(data.get('madx_survey', '')))

        if 'emit_type' in data:
            is_norm = str(data['emit_type']).lower() == 'normalized'
            self.w_emit_norm.setChecked(is_norm)
            self.w_emit_geo.setChecked(not is_norm)

        if 'panels' in data:
            loaded = [dict(p) for p in data['panels']]
            if loaded: self._panels = loaded; self._render_panel_list()

        if 'compare_files' in data:
            self._compare_files = list(data['compare_files'])
            self._render_compare_list()

        if 'elem_colors' in data:
            from core.utils import _ELEM_COLOR_DEFAULTS, _ELEM_COLOR_FALLBACK
            self._elem_colors = dict(data['elem_colors'])
            for key, btn in self._color_btns.items():
                color = self._elem_colors.get(key, _ELEM_COLOR_DEFAULTS.get(key, _ELEM_COLOR_FALLBACK))
                self._set_swatch(btn, color)

        if 'shared_xaxis' in data:
            self.w_shared_xaxis.setChecked(bool(data['shared_xaxis']))
        if 'cell_srange_on' in data:
            self.w_cell_srange.setChecked(bool(data['cell_srange_on']))
        if 'grid_enabled' in data:
            self._grid_enabled = bool(data['grid_enabled'])
            self.w_grid_enable.setChecked(self._grid_enabled)
        if 'grid_size' in data:
            self.w_grid_size.setText(str(data['grid_size']))
            self._on_grid_size_changed(data['grid_size'])

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
