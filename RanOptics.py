#!/usr/bin/env python3
# =============================================================================
# RanOptics — Accelerator Optics Plotter  (PySide6 edition)  v1.2.0
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
    win = LuxV4GUI()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
