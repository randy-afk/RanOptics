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
    # Darkened from #059669: white button text on the original scored 3.77,
    # below the 4.5 needed for body text. Same hue, now 4.95.
    ACCENT   = "#05805a",
    ACCENTH  = "#046b4d",
    ACCENT2  = "#8a5d09",
    AINK     = "#ffffff",
    ASOFT    = "rgba(5,128,90,.11)",
    COPPER   = "#d9581b",
    CSOFT    = "rgba(217,88,27,.13)",
    ERROR    = "#c4503e",
    WARN     = "#d9581b",
    RAN_CLR  = "#05805a",
    SHADOW   = "0 8px 30px rgba(60,80,60,.16)",
)

# ── Alternate dark palettes ───────────────────────────────────────────────────
# Every palette assigns the same four roles, which is what keeps a multi-colour
# theme legible instead of noisy:
#   ACCENT  — primary action (Run, focus, checked state). Never decorative.
#   COPPER  — secondary action (Dry Run, header rule).
#   ACCENT2 — section labels and log header. Saturated but only ever small caps.
#   WARN / ERROR — reserved for actual problems.
# RAN_CLR tracks ACCENT in every theme so the logo always belongs to the palette.

_PETROL = dict(
    BG       = "#0b1618",   MANTLE   = "#081113",   CRUST    = "#081113",
    PANEL    = "#112124",   PANEL2   = "#16292d",   SURFACE2 = "#1c333a",
    BORDER   = "#1f3a3f",
    FG       = "#e2eff0",   FG_DIM   = "#8fa8ab",   FG_LBL   = "#64807f",
    ACCENT   = "#22e39c",   ACCENTH  = "#4fecb3",   ACCENT2  = "#ffcb5c",
    AINK     = "#04150d",
    ASOFT    = "rgba(34,227,156,.14)",
    COPPER   = "#ff8f6b",
    CSOFT    = "rgba(255,143,107,.15)",
    ERROR    = "#f4626d",   WARN     = "#ffcb5c",   RAN_CLR  = "#22e39c",
    SHADOW   = "0 8px 30px rgba(0,0,0,.45)",
)

_SULFUR = dict(
    BG       = "#0c1226",   MANTLE   = "#080d1c",   CRUST    = "#080d1c",
    PANEL    = "#131b33",   PANEL2   = "#18213d",   SURFACE2 = "#1f2a4a",
    BORDER   = "#223052",
    FG       = "#e3ebfa",   FG_DIM   = "#90a0c0",   FG_LBL   = "#66779c",
    ACCENT   = "#e8e14f",   ACCENTH  = "#f2ec7a",   ACCENT2  = "#5fb8d3",
    AINK     = "#1a1900",
    ASOFT    = "rgba(232,225,79,.14)",
    COPPER   = "#ff8f6b",
    CSOFT    = "rgba(255,143,107,.15)",
    ERROR    = "#ff6480",   WARN     = "#ffb454",   RAN_CLR  = "#e8e14f",
    SHADOW   = "0 8px 30px rgba(0,0,0,.45)",
)

_ULTRAVIOLET = dict(
    BG       = "#150d1f",   MANTLE   = "#100819",   CRUST    = "#100819",
    PANEL    = "#1e1329",   PANEL2   = "#251831",   SURFACE2 = "#2e1f3d",
    BORDER   = "#342244",
    FG       = "#ece4f5",   FG_DIM   = "#a795b8",   FG_LBL   = "#7a6a8a",
    ACCENT   = "#b8f24f",   ACCENTH  = "#c9f57a",   ACCENT2  = "#ff8fd0",
    AINK     = "#0d1400",
    ASOFT    = "rgba(184,242,79,.14)",
    COPPER   = "#6ec8ff",
    CSOFT    = "rgba(110,200,255,.15)",
    ERROR    = "#ff5c7a",   WARN     = "#ffb454",   RAN_CLR  = "#b8f24f",
    SHADOW   = "0 8px 30px rgba(0,0,0,.5)",
)

_OXBLOOD = dict(
    BG       = "#1a1012",   MANTLE   = "#140c0e",   CRUST    = "#140c0e",
    PANEL    = "#241619",   PANEL2   = "#2b1b1e",   SURFACE2 = "#362227",
    BORDER   = "#3d262a",
    FG       = "#f2e6e8",   FG_DIM   = "#b09499",   FG_LBL   = "#8a6d73",
    ACCENT   = "#4fe0b0",   ACCENTH  = "#7ae9c6",   ACCENT2  = "#dba45f",
    AINK     = "#05140f",
    ASOFT    = "rgba(79,224,176,.14)",
    COPPER   = "#ff9ea8",
    CSOFT    = "rgba(255,158,168,.15)",
    ERROR    = "#ff5c6a",   WARN     = "#dba45f",   RAN_CLR  = "#4fe0b0",
    SHADOW   = "0 8px 30px rgba(0,0,0,.45)",
)

# ── Light siblings ────────────────────────────────────────────────────────────
# In light mode the GROUND TINT carries the theme's identity, not the accent.
# Dark accents all converge toward the same muddy midtones once they're darkened
# enough for white text to sit on them, so a light palette built only by
# darkening accents ends up indistinguishable from every other one. Panels are
# deliberately tinted rather than pure white for the same reason.

_PETROL_LIGHT = dict(
    BG       = "#dfeef0",   MANTLE   = "#cbe4e6",   CRUST    = "#cbe4e6",
    PANEL    = "#f4fbfb",   PANEL2   = "#e9f5f6",   SURFACE2 = "#d3e9eb",
    BORDER   = "#a9cdd0",
    FG       = "#062225",   FG_DIM   = "#43676a",   FG_LBL   = "#71969a",
    ACCENT   = "#00795a",   ACCENTH  = "#005f47",   ACCENT2  = "#8a5f00",
    AINK     = "#ffffff",
    ASOFT    = "rgba(0,121,90,.13)",
    COPPER   = "#c04a22",
    CSOFT    = "rgba(192,74,34,.13)",
    ERROR    = "#c0392b",   WARN     = "#8a5f00",   RAN_CLR  = "#00795a",
    SHADOW   = "0 8px 30px rgba(20,60,62,.17)",
)

_SULFUR_LIGHT = dict(
    BG       = "#dbe6f7",   MANTLE   = "#c6d7ef",   CRUST    = "#c6d7ef",
    PANEL    = "#f4f8ff",   PANEL2   = "#e8eefb",   SURFACE2 = "#cfdcf3",
    BORDER   = "#a6bcdd",
    FG       = "#0f1730",   FG_DIM   = "#4a577a",   FG_LBL   = "#7784a6",
    ACCENT   = "#6b6000",   ACCENTH  = "#544c00",   ACCENT2  = "#1c6b85",
    AINK     = "#ffffff",
    ASOFT    = "rgba(107,96,0,.13)",
    COPPER   = "#c0512a",
    CSOFT    = "rgba(192,81,42,.13)",
    ERROR    = "#b3243c",   WARN     = "#8a6a00",   RAN_CLR  = "#6b6000",
    SHADOW   = "0 8px 30px rgba(30,45,90,.16)",
)

_ULTRAVIOLET_LIGHT = dict(
    BG       = "#f4e9fb",   MANTLE   = "#e9d8f4",   CRUST    = "#e9d8f4",
    PANEL    = "#fdfaff",   PANEL2   = "#f9f2fd",   SURFACE2 = "#eeddf8",
    BORDER   = "#d0b6e6",
    FG       = "#1d0f30",   FG_DIM   = "#5b4873",   FG_LBL   = "#8977a1",
    ACCENT   = "#6a2fa8",   ACCENTH  = "#52237f",   ACCENT2  = "#476d00",
    AINK     = "#ffffff",
    ASOFT    = "rgba(106,47,168,.12)",
    COPPER   = "#0f6fa8",
    CSOFT    = "rgba(15,111,168,.13)",
    ERROR    = "#c02a55",   WARN     = "#8a5a00",   RAN_CLR  = "#6a2fa8",
    SHADOW   = "0 8px 30px rgba(60,30,90,.17)",
)

_OXBLOOD_LIGHT = dict(
    BG       = "#f5e8ea",   MANTLE   = "#ebd5d9",   CRUST    = "#ebd5d9",
    PANEL    = "#fffbfc",   PANEL2   = "#f9eef0",   SURFACE2 = "#f0dbdf",
    BORDER   = "#d9b6bc",
    FG       = "#2a0f13",   FG_DIM   = "#6d484e",   FG_LBL   = "#987076",
    ACCENT   = "#00706f",   ACCENTH  = "#005555",   ACCENT2  = "#8a5a12",
    AINK     = "#ffffff",
    ASOFT    = "rgba(0,112,111,.13)",
    COPPER   = "#b03a4a",
    CSOFT    = "rgba(176,58,74,.13)",
    ERROR    = "#b3122b",   WARN     = "#8a5a12",   RAN_CLR  = "#00706f",
    SHADOW   = "0 8px 30px rgba(80,35,40,.16)",
)

# theme name -> {mode: palette}
THEMES = {
    "Petrol":        {"dark": _PETROL,      "light": _PETROL_LIGHT},
    "Classic Green": {"dark": _DARK,        "light": _LIGHT},
    "Sulfur Sea":    {"dark": _SULFUR,      "light": _SULFUR_LIGHT},
    "Ultraviolet":   {"dark": _ULTRAVIOLET, "light": _ULTRAVIOLET_LIGHT},
    "Oxblood":       {"dark": _OXBLOOD,     "light": _OXBLOOD_LIGHT},
}
DEFAULT_THEME = "Petrol"

# Current theme mode
_current_mode = "dark"
_current_theme = DEFAULT_THEME

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

def apply_theme(name=None, mode=None):
    """Load a palette, selected by theme name and light/dark mode.

    Either argument may be omitted to keep the current value, so
    apply_theme('Petrol') restyles without changing mode and
    apply_theme(mode='light') flips mode without changing theme. The legacy
    single-argument form apply_theme('dark') / apply_theme('light') is still
    accepted, since the header toggle calls it that way.

    Unknown names fall back to the default instead of raising: this value comes
    from a user-editable settings file and must never stop the app starting.
    """
    import sys
    m = sys.modules[__name__]
    if name in ("dark", "light"):        # legacy: apply_theme('dark')
        name, mode = None, name
    name = name or getattr(m, "_current_theme", DEFAULT_THEME)
    mode = mode or getattr(m, "_current_mode", "dark")
    if name not in THEMES:
        name = DEFAULT_THEME
    if mode not in ("dark", "light"):
        mode = "dark"
    m._current_theme = name
    m._current_mode = mode
    _load(THEMES[name][mode])

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_MAIN  = QFont("IBM Plex Sans"); FONT_MAIN.setPointSize(11)
FONT_BOLD  = QFont("IBM Plex Sans"); FONT_BOLD.setPointSize(11); FONT_BOLD.setBold(True)
FONT_SMALL = QFont("IBM Plex Sans"); FONT_SMALL.setPointSize(9)
FONT_MONO  = QFont("IBM Plex Mono"); FONT_MONO.setPointSize(10)
FONT_HDR   = QFont("IBM Plex Sans"); FONT_HDR.setPointSize(18); FONT_HDR.setBold(True)
FONT_SEC   = QFont("IBM Plex Sans"); FONT_SEC.setPointSize(11); FONT_SEC.setBold(True)

# ── Load dark theme by default ────────────────────────────────────────────────
apply_theme("dark")
