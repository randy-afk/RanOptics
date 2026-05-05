#!/usr/bin/env python3
# =============================================================================
# RanOptics — Accelerator Optics Plotter  (PySide6 edition)  v1.2.1
#
# Copyright (c) 2026 Randika Gamage
# Jefferson Lab (JLab), Newport News, VA
# Licensed under the MIT License. See LICENSE file for details.
#
# Backends: Tao (Bmad), ELEGANT, xsuite, MAD-X
# GUI:      PySide6
# Contact:  randika@jlab.org
#
# Usage:
#     python RanOptics.py
#
# Requirements:
#     pip install numpy plotly PySide6
#     pip install pytao              # Tao backend
#     elegant + sddsconvert          # on PATH, for ELEGANT backend
#     pip install xsuite             # xsuite backend
#     pip install kaleido            # optional: PNG/PDF export
# =============================================================================

import sys
from PySide6.QtWidgets import QApplication
from core.gui import LuxV4GUI

def main():
    app = QApplication(sys.argv)
    from core.themes import BG, FG, FG_DIM, PANEL, BORDER, SURFACE2, ACCENT, MANTLE, CRUST
    app.setStyleSheet(f"""
        QDialog, QFileDialog {{
            background: {BG}; color: {FG};
        }}
        QFileDialog QWidget {{
            background: {BG}; color: {FG};
        }}
        QFileDialog QListView, QFileDialog QTreeView {{
            background: {MANTLE}; color: {FG};
            border: 1px solid {BORDER}; border-radius: 4px;
        }}
        QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {{
            background: {ACCENT}; color: {CRUST};
        }}
        QFileDialog QLineEdit {{
            background: {MANTLE}; color: {FG};
            border: 1px solid {BORDER}; border-radius: 4px; padding: 4px;
        }}
        QFileDialog QPushButton {{
            background: {PANEL}; color: {FG};
            border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 12px;
        }}
        QFileDialog QPushButton:hover {{ background: {SURFACE2}; }}
        QFileDialog QComboBox {{
            background: {MANTLE}; color: {FG};
            border: 1px solid {BORDER}; border-radius: 4px; padding: 4px;
        }}
        QFileDialog QComboBox QAbstractItemView {{
            background: {PANEL}; color: {FG};
        }}
        QFileDialog QLabel {{ color: {FG}; background: transparent; }}
        QFileDialog QHeaderView::section {{
            background: {PANEL}; color: {FG_DIM};
            border: none; padding: 4px;
        }}
        QFileDialog QSplitter {{ background: {BG}; }}
        QFileDialog QSideBar, QFileDialog QSidebar {{
            background: {MANTLE}; color: {FG};
        }}
    """)
    win = LuxV4GUI()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()