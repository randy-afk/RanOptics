# =============================================================================
# core/themes.py — RanOptics color palette, fonts, and stylesheets
# Supports dark (default) and light themes via apply_theme(mode).
# =============================================================================

from PySide6.QtGui import QFont

# ── Theme palettes ────────────────────────────────────────────────────────────
_DARK = dict(
    BG       = "#0e130f",   # Main background
    MANTLE   = "#0a0e0b",   # Header / chrome
    CRUST    = "#0a0e0b",   # Menubar / status bar
    PANEL    = "#141b15",   # Card / panel surface
    PANEL2   = "#1a221b",   # Slightly lighter panel
    SURFACE2 = "#222e24",   # Hover state
    BORDER   = "#2a382c",   # Borders and dividers
    FG       = "#e7eee8",   # Primary text
    FG_DIM   = "#95a797",   # Dimmed / hint text
    FG_LBL   = "#65786a",   # Label / faint text
    ACCENT   = "#22e39c",   # Emerald green — primary accent (vivid)
    ACCENTH  = "#4eecb3",   # Accent hover
    ACCENT2  = "#f0c052",   # Gold — tertiary accent (section labels/headers)
    AINK     = "#05140c",   # Text on accent bg
    ASOFT    = "rgba(34,227,156,.14)",  # Soft accent bg
    COPPER   = "#ff9a4d",   # Amber-copper — secondary accent (vivid)
    CSOFT    = "rgba(255,154,77,.15)",  # Soft copper bg
    ERROR    = "#e0705f",   # Danger / error
    WARN     = "#ff9a4d",   # Warning (copper)
    RAN_CLR  = "#22e39c",   # Brand green (same as ACCENT in new theme)
    SHADOW   = "0 8px 30px rgba(0,0,0,.45)",
)

_LIGHT = dict(
    BG       = "#e9efe7",
    MANTLE   = "#dde6db",
    CRUST    = "#dde6db",
    PANEL    = "#ffffff",
    PANEL2   = "#f4f8f3",
    SURFACE2 = "#e1e9df",
    BORDER   = "#cfdace",
    FG       = "#15201a",
    FG_DIM   = "#566656",
    FG_LBL   = "#859786",
    ACCENT   = "#059669",
    ACCENTH  = "#047a56",
    ACCENT2  = "#96650a",
    AINK     = "#ffffff",
    ASOFT    = "rgba(5,150,105,.11)",
    COPPER   = "#d9581b",
    CSOFT    = "rgba(217,88,27,.13)",
    ERROR    = "#c4503e",
    WARN     = "#d9581b",
    RAN_CLR  = "#059669",
    SHADOW   = "0 8px 30px rgba(60,80,60,.16)",
)

# Current theme mode
_current_mode = "dark"

def _load(palette):
    """Inject palette into module globals and rebuild stylesheets."""
    import sys
    m = sys.modules[__name__]
    for k, v in palette.items():
        setattr(m, k, v)
    # Keep old aliases alive
    m.PEACH      = m.ACCENT
    m.HIGHLIGHT  = m.WARN
    m.SUCCESS    = m.RAN_CLR
    _rebuild_stylesheets(m)

def _rebuild_stylesheets(m):
    BG=m.BG; MANTLE=m.MANTLE; CRUST=m.CRUST; PANEL=m.PANEL
    SURFACE2=m.SURFACE2; BORDER=m.BORDER
    FG=m.FG; FG_DIM=m.FG_DIM; FG_LBL=m.FG_LBL
    ACCENT=m.ACCENT; AINK=m.AINK; ERROR=m.ERROR

    m._ENTRY_SS = f"""
        QLineEdit {{
            background: {MANTLE}; border: 1px solid {BORDER};
            border-radius: 7px; color: {FG}; padding: 6px 10px;
            font-family: 'IBM Plex Mono', monospace;
            selection-background-color: {ACCENT}; selection-color: {AINK};
        }}
        QLineEdit:focus {{
            border-color: {ACCENT};
            border-left: 3px solid {ACCENT};
            background: {BG};
        }}
        QLineEdit[readOnly="true"] {{ color: {FG_DIM}; background: {PANEL}; }}
    """

    m._COMBO_SS = f"""
        QComboBox {{
            background: {MANTLE}; border: 1px solid {BORDER};
            border-radius: 7px; color: {FG}; padding: 5px 10px;
        }}
        QComboBox:focus {{ border-color: {ACCENT}; }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox::down-arrow {{ width: 0; height: 0; }}
        QComboBox QAbstractItemView {{
            background: {PANEL}; color: {FG}; border: 1px solid {BORDER};
            border-radius: 6px; padding: 2px;
            selection-background-color: {ACCENT}; selection-color: {AINK};
            outline: none;
        }}
    """

    m._BTN_SS = f"""
        QPushButton {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 7px; color: {ACCENT}; padding: 5px 10px;
        }}
        QPushButton:hover  {{
            background: {SURFACE2}; border-color: {ACCENT};
        }}
        QPushButton:pressed {{ background: {BORDER}; }}
        QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
    """

    m._CHK_SS = f"""
        QCheckBox {{ color: {FG}; spacing: 7px; }}
        QCheckBox::indicator {{
            width: 15px; height: 15px; border-radius: 4px;
            border: 1px solid {SURFACE2}; background: {MANTLE};
        }}
        QCheckBox::indicator:unchecked:hover {{ border-color: {ACCENT}; }}
        QCheckBox::indicator:checked {{
            background: {ACCENT}; border-color: {ACCENT}; image: none;
        }}
    """

    m._RB_SS = f"""
        QRadioButton {{ color: {FG}; spacing: 7px; }}
        QRadioButton::indicator {{
            width: 14px; height: 14px; border-radius: 7px;
            border: 1px solid {SURFACE2}; background: {MANTLE};
        }}
        QRadioButton::indicator:checked {{
            background: {ACCENT}; border-color: {ACCENT}; border-width: 3px;
        }}
    """

    m._TAB_SS = f"""
        QTabWidget::pane {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 9px; top: -1px;
        }}
        QTabBar::tab {{
            background: {MANTLE}; color: {FG_LBL}; padding: 7px 18px;
            border: 1px solid {BORDER}; border-bottom: none; margin-right: 3px;
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            font-weight: 500;
        }}
        QTabBar::tab:selected {{
            background: {PANEL}; color: {ACCENT};
            border-bottom-color: {PANEL};
        }}
        QTabBar::tab:hover:!selected {{ background: {SURFACE2}; color: {FG}; }}
    """

    m._SCROLL_SS = f"""
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

def apply_theme(mode="dark"):
    """Switch between 'dark' and 'light' themes. Call from GUI."""
    import sys
    m = sys.modules[__name__]
    m._current_mode = mode
    _load(_DARK if mode == "dark" else _LIGHT)

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_MAIN  = QFont("IBM Plex Sans"); FONT_MAIN.setPointSize(11)
FONT_BOLD  = QFont("IBM Plex Sans"); FONT_BOLD.setPointSize(11); FONT_BOLD.setBold(True)
FONT_SMALL = QFont("IBM Plex Sans"); FONT_SMALL.setPointSize(9)
FONT_MONO  = QFont("IBM Plex Mono"); FONT_MONO.setPointSize(10)
FONT_HDR   = QFont("IBM Plex Sans"); FONT_HDR.setPointSize(18); FONT_HDR.setBold(True)
FONT_SEC   = QFont("IBM Plex Sans"); FONT_SEC.setPointSize(11); FONT_SEC.setBold(True)

# ── Load dark theme by default ────────────────────────────────────────────────
apply_theme("dark")
