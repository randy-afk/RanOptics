"""
_RanOpticsLogo — QPainter-based scalable logo widget.

Usage:
    logo = _RanOpticsLogo()                    # GUI header: 140x45
    logo = _RanOpticsLogo(size=(600, 200))     # Docs / export

Layout:
    - Y-axis at x = W * YAXIS_X_FRAC  (vertical divider, default 70%)
    - X-axis at y = H / 2              (exact vertical center)
    - 4 panels in x<0 region, each with a unique curve
    - R centered in upper-right quadrant, O centered in lower-right quadrant
    - Arrows on both axes

Tweak the constants to adjust appearance.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtGui     import (QPainter, QPen, QColor, QFont,
                                QPolygon, QPainterPath, QBrush)
from PySide6.QtCore    import Qt, QRect, QPoint, QPointF, QSize

# ── Geometry ──────────────────────────────────────────────────────────────────
YAXIS_X_FRAC    = 0.750   # y-axis x position as fraction of width
MARGIN_FRAC     = 0.07   # margin as fraction of height
PANEL_GAP_FRAC  = 0.05   # gap between panels as fraction of height
PANEL_R_GAP_FRAC= 0.08   # gap between panels and y-axis as fraction of height
AXIS_COLOR      = "#4a7a68"
ARROW_FRAC      = 0.14   # arrowhead size as fraction of height

DEFAULT_W, DEFAULT_H = 140, 45

# ── Panel definitions ─────────────────────────────────────────────────────────
PANELS = [
    {"color": "#00e676", "opacity": 0.28},  # green  — beta function
    {"color": "#FDA769", "opacity": 0.28},  # peach  — dispersion
    {"color": "#4fc3f7", "opacity": 0.28},  # blue   — orbit
    {"color": "#d62828", "opacity": 0.28},  # red    — beam size
]
PANEL_STROKE_OPACITY = 0.70

# ── Letter colors ─────────────────────────────────────────────────────────────
R_COLOR = "#00e676"
O_COLOR = "#d62828"


class _RanOpticsLogo(QWidget):
    """Scalable RanOptics logo. All geometry derived from widget size."""

    def __init__(self, parent=None, size=(DEFAULT_W, DEFAULT_H)):
        super().__init__(parent)
        self._w, self._h = size
        self.setFixedSize(self._w, self._h)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

    def sizeHint(self):
        return QSize(self._w, self._h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W = self.width()
        H = self.height()

        # Derived geometry
        margin   = max(2, int(H * MARGIN_FRAC))
        gap      = max(1, int(H * PANEL_GAP_FRAC))
        rgap     = max(2, int(H * PANEL_R_GAP_FRAC))
        arrow    = max(3, int(H * ARROW_FRAC))
        yaxis_x  = int(W * YAXIS_X_FRAC)
        xaxis_y  = H // 2
        panel_w  = yaxis_x - margin - rgap
        n        = len(PANELS)
        panel_h  = (H - 2 * margin - gap * (n - 1)) / n
        axis_w   = max(1.0, H / DEFAULT_H * 1.2)

        # ── Panels ────────────────────────────────────────────────────────────
        for i, panel in enumerate(PANELS):
            y0    = margin + i * (panel_h + gap)
            color = QColor(panel["color"])

            fill = QColor(color); fill.setAlphaF(panel["opacity"])
            stroke = QColor(color); stroke.setAlphaF(PANEL_STROKE_OPACITY)

            p.setBrush(QBrush(fill))
            p.setPen(QPen(stroke, 0.8))
            p.drawRoundedRect(QRect(margin, int(y0), panel_w, int(panel_h)), 2, 2)

            # curve
            p.setPen(QPen(color, max(1.0, axis_w * 0.9), Qt.SolidLine,
                          Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)

            x0 = margin + max(2, int(panel_w * 0.03))
            x1 = margin + panel_w - max(3, int(panel_w * 0.05))
            yc = y0 + panel_h / 2
            a  = panel_h * 0.40
            pw = x1 - x0

            path = QPainterPath()
            if i == 0:
                # Beta — sine wave up-down-up
                path.moveTo(x0, yc)
                path.cubicTo(x0+pw*0.12, yc-a,   x0+pw*0.25, yc-a,   x0+pw*0.38, yc)
                path.cubicTo(x0+pw*0.50, yc+a,   x0+pw*0.63, yc+a,   x0+pw*0.75, yc)
                path.cubicTo(x0+pw*0.85, yc-a*0.5, x0+pw*0.94, yc-a*0.5, x1, yc-a*0.3)
            elif i == 1:
                # Dispersion — smooth S-curve
                path.moveTo(x0, yc+a*0.6)
                path.cubicTo(x0+pw*0.25, yc+a*0.6, x0+pw*0.40, yc-a*0.6, x0+pw*0.55, yc-a*0.6)
                path.cubicTo(x0+pw*0.75, yc-a*0.6, x0+pw*0.90, yc-a*0.9, x1, yc-a*0.9)
            elif i == 2:
                # Orbit — flat with small dip and recovery
                path.moveTo(x0, yc-a*0.1)
                path.cubicTo(x0+pw*0.20, yc-a*0.1, x0+pw*0.30, yc+a*0.65, x0+pw*0.45, yc+a*0.65)
                path.cubicTo(x0+pw*0.60, yc+a*0.65, x0+pw*0.78, yc-a*0.1, x1, yc-a*0.1)
            elif i == 3:
                # Beam size — gradual growth
                path.moveTo(x0, yc+a*0.3)
                path.cubicTo(x0+pw*0.30, yc+a*0.1, x0+pw*0.60, yc-a*0.4, x0+pw*0.82, yc-a*0.8)
                path.cubicTo(x0+pw*0.92, yc-a*0.95, x1-2, yc-a, x1, yc-a)

            p.drawPath(path)

        # ── Axes ──────────────────────────────────────────────────────────────
        ac  = QColor(AXIS_COLOR)
        pen = QPen(ac, axis_w)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.setBrush(QBrush(ac))

        # x axis
        x_end = W - margin - 1
        p.drawLine(QPointF(margin, xaxis_y), QPointF(x_end - arrow, xaxis_y))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygon([
            QPoint(x_end,         xaxis_y),
            QPoint(x_end - arrow, xaxis_y - arrow // 2),
            QPoint(x_end - arrow, xaxis_y + arrow // 2),
        ]))

        # y axis
        p.setPen(pen)
        y_top = margin
        p.drawLine(QPointF(yaxis_x, y_top + arrow), QPointF(yaxis_x, H - margin))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygon([
            QPoint(yaxis_x,           y_top),
            QPoint(yaxis_x - arrow//2, y_top + arrow),
            QPoint(yaxis_x + arrow//2, y_top + arrow),
        ]))

        # ── R and O — centered in their quadrant halves ───────────────────────
        lx = yaxis_x + max(2, int(W * 0.015))
        lw = W - lx - margin

        font_size = max(6, int((xaxis_y - margin) * 0.88))
        font = QFont()
        font.setWeight(QFont.Weight.Bold)
        font.setPixelSize(font_size)
        p.setFont(font)

        # R — vertically centered in upper half
        p.setPen(QPen(QColor(R_COLOR)))
        p.drawText(QRect(lx, margin, lw, xaxis_y - margin),
                   Qt.AlignLeft | Qt.AlignVCenter, "R")

        # O — vertically centered in lower half
        p.setPen(QPen(QColor(O_COLOR)))
        p.drawText(QRect(lx, xaxis_y, lw, H - xaxis_y - margin),
                   Qt.AlignLeft | Qt.AlignVCenter, "O")

        p.end()
