# =============================================================================
# core/themes.py — RanOptics color palette, fonts, and stylesheets
# To retheme: edit only the color block. All UI colors derive from these.
# =============================================================================

from PySide6.QtGui import QFont

# ── Color Palette ─────────────────────────────────────────────────────────────
BG       = "#2C5446"   # Main background — deep dark teal
MANTLE   = "#234038"   # Deeper background (input fields, log area)
CRUST    = "#1a2f28"   # Darkest (menubar, status bar)
PANEL    = "#3D6B5C"   # Card / panel surface
SURFACE2 = "#4A7D6C"   # Hover state
BORDER   = "#5A8A78"   # Borders and dividers
FG       = "#EEF5F2"   # Primary text
FG_DIM   = "#A8C4BC"   # Dimmed / hint text
FG_LBL   = "#8AB0A6"   # Label text
ACCENT   = "#FDA769"   # Peach — UI highlights, active states
RAN_CLR  = "#00e676"   # Bright green — RanOptics "Ran" logo
ERROR    = "#d62828"   # Bright red — DO NOT CHANGE
ACCENT2  = "#FDA769"   # Peach / orange accent
WARN     = "#FEC868"   # Warm yellow
SUCCESS  = "#00e676"   # Bright green — success state
TEAL     = "#FEC868"   # Info (warm yellow)
PEACH    = "#FDA769"   # Peach (alias for ACCENT2)

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_MAIN  = QFont(); FONT_MAIN.setPointSize(11)
FONT_BOLD  = QFont(); FONT_BOLD.setPointSize(11);  FONT_BOLD.setBold(True)
FONT_SMALL = QFont(); FONT_SMALL.setPointSize(11)
FONT_MONO  = QFont("Monospace"); FONT_MONO.setPointSize(10)
FONT_HDR   = QFont("Monospace"); FONT_HDR.setPointSize(16); FONT_HDR.setBold(True)
FONT_SEC   = QFont(); FONT_SEC.setPointSize(11);   FONT_SEC.setBold(True)

# ── Stylesheets ───────────────────────────────────────────────────────────────
_ENTRY_SS = f"""
    QLineEdit {{
        background: {MANTLE}; border: 1px solid {BORDER};
        border-radius: 8px; color: {FG}; padding: 4px 10px;
        selection-background-color: {ACCENT}; selection-color: {CRUST};
    }}
    QLineEdit:focus {{
        border-color: {ACCENT};
        border-left: 3px solid {ACCENT};
        background: {BG};
    }}
    QLineEdit[readOnly="true"] {{ color: {FG_DIM}; background: {PANEL}; }}
"""

_COMBO_SS = f"""
    QComboBox {{
        background: {MANTLE}; border: 1px solid {BORDER};
        border-radius: 8px; color: {FG}; padding: 4px 10px;
    }}
    QComboBox:focus {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ width: 0; height: 0; }}
    QComboBox QAbstractItemView {{
        background: {PANEL}; color: {FG}; border: 1px solid {BORDER};
        border-radius: 6px; padding: 2px;
        selection-background-color: {ACCENT}; selection-color: {CRUST};
        outline: none;
    }}
"""

_BTN_SS = f"""
    QPushButton {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 8px; color: {ACCENT}; padding: 4px 10px;
        font-weight: 500;
    }}
    QPushButton:hover  {{
        background: {SURFACE2}; border-color: {ACCENT}; color: {ACCENT};
    }}
    QPushButton:pressed {{ background: {BORDER}; }}
    QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
"""

_CHK_SS = f"""
    QCheckBox {{ color: {FG}; spacing: 7px; }}
    QCheckBox::indicator {{
        width: 15px; height: 15px; border-radius: 4px;
        border: 1px solid {SURFACE2}; background: {MANTLE};
    }}
    QCheckBox::indicator:unchecked:hover {{ border-color: {ACCENT}; }}
    QCheckBox::indicator:checked {{
        background: {ACCENT}; border-color: {ACCENT};
        image: none;
    }}
"""

_RB_SS = f"""
    QRadioButton {{ color: {FG}; spacing: 7px; }}
    QRadioButton::indicator {{
        width: 14px; height: 14px; border-radius: 7px;
        border: 1px solid {SURFACE2}; background: {MANTLE};
    }}
    QRadioButton::indicator:checked {{
        background: {ACCENT}; border-color: {ACCENT};
        border-width: 3px;
    }}
"""

_TAB_SS = f"""
    QTabWidget::pane {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 10px; top: -1px;
    }}
    QTabBar::tab {{
        background: {MANTLE}; color: {FG_LBL}; padding: 7px 20px;
        border: 1px solid {BORDER}; border-bottom: none; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        background: {PANEL}; color: {ACCENT};
        border-bottom-color: {PANEL};
    }}
    QTabBar::tab:hover:!selected {{ background: {SURFACE2}; color: {FG}; }}
"""

_SCROLL_SS = f"""
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: {MANTLE}; width: 6px; margin: 0; border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {SURFACE2}; border-radius: 3px; min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
