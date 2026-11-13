"""The pySEBpolymers designer: draw structures instead of writing code.

Palette (left)   sub-units, patterns (Array, Chain, Generations) and the components of the design; drag one onto
                 the canvas or double-click it
Canvas (centre)  nodes with their reference points as ports along the edge (round: one point, diamond: a
                 distributed point such as a surface or contour). Drag from a port to another port to link them;
                 a link that would close a loop is refused. Wheel: zoom, middle button: pan, left drag on the empty
                 canvas: select. Del deletes, Ctrl+C / Ctrl+V / Ctrl+D copy, paste, duplicate, Ctrl+Z / Ctrl+Y
                 undo, redo, Ctrl+A selects all
Inspector (right) name, tag and type of the selected node; element, counts and points of a pattern; the labels of
                 a link at distributed points (the same label on the same node is the same point)

Components are designs of their own (Editing: …), with an in-port and an out-port (right-click a port of a
sub-unit); patterns repeat them. Every valid change emits :attr:`DesignerWidget.changed`.
"""
from __future__ import annotations

import json
import math
import re

from PySide6 import QtCore, QtGui, QtWidgets

from .graph import PATTERNS, UNIT_KINDS, Design, node_width, short_name
from .subunits import REGISTRY

__all__ = ["DesignerWidget", "DesignView", "DesignScene"]

MIME = "application/x-pysebpolymers-item"
MIME_NODES = "application/x-pysebpolymers-nodes"

CATEGORY = {"GaussianPolymer": "polymer", "GaussianLoop": "polymer", "FloryPolymer": "polymer", "FloryLoop": "polymer",
            "ExcludedVolumePolymer": "polymer", "WormLikeChain": "worm", "WormLikeChainEV": "worm", "ThinRod": "thin", "ThinCircle": "thin",
            "ThinDisk": "thin", "ThinSphericalShell": "thin", "SolidSphere": "solid", "SolidSphericalShell": "solid",
            "SolidCylinder": "solid", "Spheroid": "solid", "Point": "point"}
COLOURS = {"polymer": "#6B8E23", "worm": "#4F7A3A", "thin": "#2A8C82", "solid": "#3A6EA5", "point": "#6E7781", "pattern": "#B7791F",
           "component": "#8E5BB5"}
TIPS = {
    "Point": "massless junction point (no scattering)", "GaussianPolymer": "Gaussian chain (Rg)",
    "GaussianLoop": "Gaussian ring (Rg)",
    "FloryPolymer": "chain with Flory exponent nu (Rg, nu): 0.5 ideal, 0.588 good solvent, 1/3 collapsed",
    "FloryLoop": "ring with Flory exponent nu (Rg, nu)",
    "ExcludedVolumePolymer": "excluded-volume chain with non-Gaussian (des Cloizeaux) distances (Rg, nu)",
    "WormLikeChain": "semiflexible chain, Kholodenko's worm (contour length L, Kuhn length b)",
    "WormLikeChainEV": "semiflexible chain with excluded volume, Pedersen-Schurtenberger (L, Kuhn length b)", "ThinRod": "infinitely thin rod (L)", "SolidSphere": "solid sphere (R)",
    "ThinSphericalShell": "infinitely thin spherical shell (R)", "SolidSphericalShell": "solid shell (Ri, Ro)",
    "ThinCircle": "thin circular ring (R)", "ThinDisk": "infinitely thin disk (R)",
    "SolidCylinder": "solid cylinder (R, L)", "Spheroid": "ellipsoid of revolution (a, c)",
    "Array": "n copies of an element, attached at one point of a host (independent random points of a distributed "
             "point, or all at the same point)",
    "Chain": "n copies of an element joined out -> in; ports start and end",
    "Generations": "dendrimer: f elements at the root, f-1 new ones at every end, g generations",
}


def _colour_of(design, node):
    if node.is_unit:
        return QtGui.QColor(COLOURS[CATEGORY.get(node.kind, "point")])
    return QtGui.QColor(COLOURS["component" if node.element_is_component else "pattern"])


def _glyph(p, kind, rect, colour):
    """Small drawing of a sub-unit type."""
    p.save()
    pen = QtGui.QPen(colour, 1.6)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    c = rect.center()
    r = min(rect.width(), rect.height()) / 2 - 1
    if kind in ("WormLikeChain", "WormLikeChainEV"):
        path = QtGui.QPainterPath(QtCore.QPointF(rect.left(), rect.bottom() - 1))
        path.cubicTo(rect.left() + rect.width() * 0.4, rect.top() - r * 0.6, rect.left() + rect.width() * 0.6,
                     rect.bottom() + r * 0.6, rect.right(), rect.top() + 1)
        p.drawPath(path)
    elif kind in ("GaussianPolymer", "FloryPolymer", "ExcludedVolumePolymer"):
        amp, n = (0.6, 8) if kind == "GaussianPolymer" else (0.95, 5)        # swollen: fewer, wider swings
        path = QtGui.QPainterPath(QtCore.QPointF(rect.left(), c.y()))
        for i in range(1, n + 1):
            x = rect.left() + rect.width() * i / n
            y = c.y() + (r * amp if i % 2 else -r * amp)
            path.lineTo(x, y)
        p.drawPath(path)
    elif kind in ("GaussianLoop", "FloryLoop"):
        path = QtGui.QPainterPath()
        wig = 0.18 if kind == "GaussianLoop" else 0.08
        for i in range(25):
            t = 2 * math.pi * i / 24
            rr = r * (0.75 + wig * math.sin(5 * t))
            pt = QtCore.QPointF(c.x() + rr * math.cos(t), c.y() + rr * math.sin(t))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.drawPath(path)
    elif kind == "ThinRod":
        p.drawLine(QtCore.QPointF(rect.left(), rect.bottom()), QtCore.QPointF(rect.right(), rect.top()))
    elif kind in ("SolidSphere", "Spheroid"):
        p.setBrush(colour)
        w = r if kind == "SolidSphere" else r * 1.0
        h = r if kind == "SolidSphere" else r * 0.6
        p.drawEllipse(c, w, h)
    elif kind in ("ThinSphericalShell", "ThinCircle"):
        p.drawEllipse(c, r, r)
    elif kind == "SolidSphericalShell":
        p.drawEllipse(c, r, r)
        p.drawEllipse(c, r * 0.55, r * 0.55)
    elif kind == "ThinDisk":
        p.setBrush(colour)
        p.drawEllipse(c, r, r * 0.35)
    elif kind == "SolidCylinder":
        p.setBrush(colour)
        p.drawRoundedRect(QtCore.QRectF(c.x() - r, c.y() - r * 0.5, 2 * r, r), 2, 2)
    elif kind == "Point":
        p.setBrush(colour)
        p.drawEllipse(c, r * 0.35, r * 0.35)
    else:                                   # patterns: stacked squares
        for k in range(3):
            d = 3 * k
            p.drawRect(QtCore.QRectF(rect.left() + d, rect.top() + d, rect.width() * 0.6, rect.height() * 0.6))
    p.restore()


# ====================================================================== scene items
class PortItem(QtWidgets.QGraphicsItem):
    R = 6.0

    def __init__(self, node_item, port, distributed, pos, role=""):
        super().__init__(node_item)
        self.node_item, self.port, self.distributed, self.role = node_item, port, distributed, role
        self.setPos(pos)
        self.setAcceptHoverEvents(True)
        self._hover = False
        self.setZValue(2)
        kind = node_item.node.kind if node_item.node.is_unit else node_item.node.type
        what = "distributed point (one per label)" if distributed else "point"
        self.setToolTip(f"{node_item.node.name}.{port}: {what}" + (f" — component {role}-port" if role else "")
                        + "\nDrag to another port to link.")
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        del kind

    def boundingRect(self):
        r = self.R + 2
        return QtCore.QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, p, option, widget=None):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        col = QtGui.QColor("#FFC857") if self.role == "in" else QtGui.QColor("#E4572E") if self.role == "out" \
            else QtGui.QColor("white")
        if self._hover:
            col = QtGui.QColor("#FFE08A")
        p.setBrush(col)
        p.setPen(QtGui.QPen(QtGui.QColor("#1F2933"), 1.4))
        r = self.R
        if self.distributed:
            poly = QtGui.QPolygonF([QtCore.QPointF(0, -r - 1), QtCore.QPointF(r + 1, 0), QtCore.QPointF(0, r + 1),
                                    QtCore.QPointF(-r - 1, 0)])
            p.drawPolygon(poly)
        else:
            p.drawEllipse(QtCore.QPointF(0, 0), r, r)

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()

    def scene_centre(self):
        return self.mapToScene(QtCore.QPointF(0, 0))


class NodeItem(QtWidgets.QGraphicsObject):
    moved = QtCore.Signal()

    def __init__(self, design, node, roles=None):
        super().__init__()
        self.design, self.node = design, node
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(node.x, node.y)
        self.setZValue(1)
        self.ports = {}
        ports = design.ports_of(node)
        if node.is_unit:
            self.w = node_width(design, node)
            self.h = 72.0
            for k, port in enumerate(ports):
                x = self.w * (k + 0.5) / len(ports)
                self.ports[port] = PortItem(self, port, REGISTRY[node.kind].is_distributed(port),
                                            QtCore.QPointF(x, self.h), (roles or {}).get(port, ""))
        else:
            self.w, self.h = 196.0, 76.0
            pos = {"anchor": QtCore.QPointF(0, self.h / 2), "start": QtCore.QPointF(0, self.h / 2),
                   "end": QtCore.QPointF(self.w, self.h / 2), "root": QtCore.QPointF(0, self.h / 2)}
            for port in ports:
                self.ports[port] = PortItem(self, port, False, pos[port])
        self.setToolTip(TIPS.get(node.kind if node.is_unit else node.type, ""))
        self.host_distributed = None
        for ln in design.links_of(node.id):
            o = ln.other(node.id)
            if o[0] in design.nodes:
                self.host_distributed = design.is_distributed(design.nodes[o[0]], o[1])

    def boundingRect(self):
        return QtCore.QRectF(-4, -4, self.w + 12, self.h + 12)

    def paint(self, p, option, widget=None):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        nd = self.node
        col = _colour_of(self.design, nd)
        pal = widget.palette() if widget is not None else QtWidgets.QApplication.palette()
        dark = pal.color(QtGui.QPalette.ColorRole.Window).lightness() < 128
        body = QtGui.QColor(col)
        body.setAlpha(55 if dark else 38)
        base = pal.color(QtGui.QPalette.ColorRole.Base)
        rect = QtCore.QRectF(0, 0, self.w, self.h)
        if not nd.is_unit:                  # stack of copies behind
            for k in (2, 1):
                p.setPen(QtGui.QPen(col, 1))
                p.setBrush(base)
                p.drawRoundedRect(rect.translated(4 * k, 4 * k), 8, 8)
        p.setBrush(base)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, 8, 8)
        p.setBrush(body)
        sel = self.isSelected()
        p.setPen(QtGui.QPen(QtGui.QColor("#FFB000") if sel else col, 2.6 if sel else 1.4))
        p.drawRoundedRect(rect, 8, 8)
        head = QtCore.QRectF(0, 0, self.w, 22)
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.FillRule.WindingFill)
        path.addRoundedRect(head, 8, 8)
        path.addRect(QtCore.QRectF(0, 12, self.w, 10))
        p.fillPath(path.simplified(), col)
        p.setPen(QtGui.QColor("white"))
        f = p.font()
        f.setBold(True)
        f.setPointSizeF(max(7.0, f.pointSizeF()))
        p.setFont(f)
        kind = nd.kind if nd.is_unit else nd.type
        p.drawText(head.adjusted(24, 0, -6, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, nd.name)
        _glyph(p, kind, QtCore.QRectF(5, 4, 15, 14), QtGui.QColor("white"))
        f.setBold(False)
        f.setPointSizeF(max(6.5, f.pointSizeF() - 1))
        p.setFont(f)
        p.setPen(pal.color(QtGui.QPalette.ColorRole.Text))
        if nd.is_unit:
            line = f"{nd.kind}   tag {nd.tag}"
            flags = [w for w, f in (("polydisperse", nd.poly), ("thick", nd.thick)) if f]
            if flags:
                line += "\n" + ", ".join(flags)
        else:
            elem = nd.element + (" ⟨component⟩" if nd.element_is_component else "")
            if nd.type == "Array":
                where = "at one point" if self.host_distributed is False else \
                    ("independent points" if nd.mode == "independent" else "same point")
                line = f"{nd.n} × {elem}\n{where}"
            elif nd.type == "Chain":
                line = f"{nd.n} × {elem} in a row"
            else:
                line = f"{nd.g} generations, f = {nd.f}\n{elem}"
            if not nd.element_is_component:
                line += f"   tag {nd.tag}" + ("+gen" if nd.type == "Generations" and nd.gen_tags else "")
            flags = [w for w, f in (("poly", nd.poly), ("thick", nd.thick), ("excl. vol.", nd.vex)) if f]
            if flags:
                line += "   [" + ", ".join(flags) + "]"
        p.drawText(QtCore.QRectF(8, 24, self.w - 16, 30), QtCore.Qt.AlignmentFlag.AlignLeft, line)
        if nd.is_unit:                      # port names above the ports
            f.setPointSizeF(max(6.0, f.pointSizeF() - 0.5))
            p.setFont(f)
            p.setPen(pal.color(QtGui.QPalette.ColorRole.PlaceholderText))
            for port, it in self.ports.items():
                x = it.pos().x()
                p.drawText(QtCore.QRectF(x - 30, self.h - 22, 60, 12), QtCore.Qt.AlignmentFlag.AlignCenter, port)
        else:
            f.setPointSizeF(max(6.0, f.pointSizeF() - 0.5))
            p.setFont(f)
            p.setPen(pal.color(QtGui.QPalette.ColorRole.PlaceholderText))
            for port, it in self.ports.items():
                al = QtCore.Qt.AlignmentFlag.AlignLeft if it.pos().x() < 1 else QtCore.Qt.AlignmentFlag.AlignRight
                x = 10 if it.pos().x() < 1 else self.w - 70
                p.drawText(QtCore.QRectF(x, self.h - 18, 60, 12), al, port)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x, self.node.y = self.pos().x(), self.pos().y()
            self.moved.emit()
        return super().itemChange(change, value)


class LinkItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, link, pa, pb):
        super().__init__()
        self.link, self.pa, self.pb = link, pa, pb
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0)
        self.labels = []
        for port, end in ((pa, link.a), (pb, link.b)):
            if port.distributed and end[2]:
                t = QtWidgets.QGraphicsSimpleTextItem("#" + end[2], self)
                t.setBrush(QtGui.QColor("#8A5A00"))
                self.labels.append((t, port))
        self.update_path()

    def update_path(self):
        a, b = self.pa.scene_centre(), self.pb.scene_centre()
        da = self._dir(self.pa)
        db = self._dir(self.pb)
        d = max(40.0, 0.4 * math.hypot(b.x() - a.x(), b.y() - a.y()))
        path = QtGui.QPainterPath(a)
        path.cubicTo(a + da * d, b + db * d, b)
        self.setPath(path)
        for t, port in self.labels:
            c = port.scene_centre() + self._dir(port) * 16
            t.setPos(c.x() + 4, c.y() - 6)

    @staticmethod
    def _dir(port):
        p = port.pos()
        ni = port.node_item
        if p.y() >= ni.h - 0.5:
            return QtCore.QPointF(0, 1)
        if p.x() <= 0.5:
            return QtCore.QPointF(-1, 0)
        return QtCore.QPointF(1, 0)

    def paint(self, p, option, widget=None):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sel = self.isSelected()
        pal = widget.palette() if widget is not None else QtWidgets.QApplication.palette()
        col = QtGui.QColor("#FFB000") if sel else pal.color(QtGui.QPalette.ColorRole.Text)
        p.setPen(QtGui.QPen(col, 3.0 if sel else 1.8))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(self.path())

    def shape(self):
        st = QtGui.QPainterPathStroker()
        st.setWidth(10)
        return st.createStroke(self.path())


class DesignScene(QtWidgets.QGraphicsScene):
    """Shows one design (the structure or a component)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.design = None
        self.node_items, self.link_items = {}, []

    def show_design(self, design):
        sel_nodes = {it.node.id for it in self.selectedItems() if isinstance(it, NodeItem)} \
            if self.design is design else set()
        self.clear()
        self.design = design
        self.node_items, self.link_items = {}, []
        roles = {}
        for which, (nid, ref, _l) in design.ports.items():
            roles.setdefault(nid, {})[ref] = which
        for nd in design.nodes.values():
            it = NodeItem(design, nd, roles.get(nd.id))
            it.moved.connect(self._node_moved)
            self.addItem(it)
            self.node_items[nd.id] = it
            if nd.id in sel_nodes:
                it.setSelected(True)
        for ln in design.links:
            ia, ib = self.node_items.get(ln.a[0]), self.node_items.get(ln.b[0])
            if ia is None or ib is None or ln.a[1] not in ia.ports or ln.b[1] not in ib.ports:
                continue
            li = LinkItem(ln, ia.ports[ln.a[1]], ib.ports[ln.b[1]])
            self.addItem(li)
            self.link_items.append(li)
        r = self.itemsBoundingRect().adjusted(-400, -300, 400, 300)
        self.setSceneRect(r.united(QtCore.QRectF(-600, -400, 1200, 800)))

    def _node_moved(self):
        for li in self.link_items:
            li.update_path()


class DesignView(QtWidgets.QGraphicsView):
    link_requested = QtCore.Signal(object, object)          # PortItem, PortItem
    dropped = QtCore.Signal(str, QtCore.QPointF)
    context = QtCore.Signal(object, QtCore.QPoint)          # item or None, global pos
    nodes_moved = QtCore.Signal()

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self._drag_port = None
        self._temp = None
        self._pan = None
        self._pressed_pos = None

    # zoom and pan
    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        z = self.transform().m11() * f
        if 0.1 < z < 4:
            self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._pan = e.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            it = self.itemAt(e.position().toPoint())
            if isinstance(it, PortItem):
                self._drag_port = it
                self._temp = QtWidgets.QGraphicsLineItem(QtCore.QLineF(it.scene_centre(), it.scene_centre()))
                self._temp.setPen(QtGui.QPen(QtGui.QColor("#FFB000"), 2, QtCore.Qt.PenStyle.DashLine))
                self.scene().addItem(self._temp)
                return
            self._pressed_pos = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._temp is not None:
            line = self._temp.line()
            line.setP2(self.mapToScene(e.position().toPoint()))
            self._temp.setLine(line)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None and e.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._pan = None
            self.unsetCursor()
            return
        if self._temp is not None:
            self.scene().removeItem(self._temp)
            self._temp = None
            src, self._drag_port = self._drag_port, None
            target = None
            for it in self.items(e.position().toPoint()):
                if isinstance(it, PortItem) and it is not src:
                    target = it
                    break
            if target is not None:
                self.link_requested.emit(src, target)
            return
        moved = self._pressed_pos is not None and (e.position() - self._pressed_pos).manhattanLength() > 3
        self._pressed_pos = None
        super().mouseReleaseEvent(e)
        if moved and any(isinstance(i, NodeItem) for i in self.scene().selectedItems()):
            self.nodes_moved.emit()

    def contextMenuEvent(self, e):
        it = self.itemAt(e.pos())
        while it is not None and not isinstance(it, (PortItem, NodeItem, LinkItem)):
            it = it.parentItem()
        self.context.emit(it, e.globalPos())

    # drops from the palette
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            what = bytes(e.mimeData().data(MIME)).decode()
            self.dropped.emit(what, self.mapToScene(e.position().toPoint()))
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def fit(self, min_zoom=0.6):
        """Show the whole design, but not smaller than min_zoom (then the view scrolls) nor larger than 1."""
        r = self.scene().itemsBoundingRect()
        if r.isValid() and not r.isEmpty():
            self.fitInView(r.adjusted(-40, -40, 40, 40), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            z = self.transform().m11()
            if z > 1.0 or z < min_zoom:
                self.resetTransform()
                zz = min(1.0, max(z, min_zoom))
                self.scale(zz, zz)
                self.centerOn(r.center() if z > 1.0 else r.topLeft() + QtCore.QPointF(
                    self.viewport().width() / (2 * zz) - 40, self.viewport().height() / (2 * zz) - 40))


class Palette(QtWidgets.QTreeWidget):
    activated_item = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.setRootIsDecorated(True)
        self.itemDoubleClicked.connect(lambda it, _c: it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                                       and self.activated_item.emit(it.data(0, QtCore.Qt.ItemDataRole.UserRole)))
        self.units = QtWidgets.QTreeWidgetItem(["Sub-units"])
        self.patterns = QtWidgets.QTreeWidgetItem(["Patterns"])
        self.comps = QtWidgets.QTreeWidgetItem(["Components"])
        for top in (self.units, self.patterns, self.comps):
            top.setFlags(top.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            self.addTopLevelItem(top)
        for k in UNIT_KINDS:
            it = QtWidgets.QTreeWidgetItem([k])
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"unit:{k}")
            it.setToolTip(0, TIPS.get(k, k))
            it.setIcon(0, self._icon(k, COLOURS[CATEGORY[k]]))
            self.units.addChild(it)
        for k in PATTERNS:
            it = QtWidgets.QTreeWidgetItem([k])
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"pattern:{k}")
            it.setToolTip(0, TIPS[k])
            it.setIcon(0, self._icon(k, COLOURS["pattern"]))
            self.patterns.addChild(it)
        self.expandAll()
        self.setMinimumWidth(150)

    @staticmethod
    def _icon(kind, colour):
        pm = QtGui.QPixmap(20, 20)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        _glyph(p, kind, QtCore.QRectF(2, 3, 16, 14), QtGui.QColor(colour))
        p.end()
        return QtGui.QIcon(pm)

    def set_components(self, names):
        self.comps.takeChildren()
        for n in names:
            it = QtWidgets.QTreeWidgetItem([n])
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"comp:{n}")
            it.setToolTip(0, f"component '{n}': drop to add an Array of it (change the type in the inspector)")
            it.setIcon(0, self._icon("Array", COLOURS["component"]))
            self.comps.addChild(it)
        self.comps.setExpanded(True)

    def startDrag(self, actions):
        # own QDrag (it takes the mime data); a Python override of mimeData() would hand Qt an object that
        # PySide6 may delete a second time
        it = self.currentItem()
        v = it.data(0, QtCore.Qt.ItemDataRole.UserRole) if it is not None else None
        if not v:
            return
        md = QtCore.QMimeData()
        md.setData(MIME, QtCore.QByteArray(v.encode()))
        drag = QtGui.QDrag(self)
        drag.setMimeData(md)
        drag.setPixmap(it.icon(0).pixmap(20, 20))
        drag.exec(QtCore.Qt.DropAction.CopyAction)


# ====================================================================== inspector
class Inspector(QtWidgets.QScrollArea):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setWidgetResizable(True)
        self.setMinimumWidth(220)
        self._w = None
        self._pending = None
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._show_pending)
        self.show_now(None)

    def _new(self, title):
        old = self.takeWidget()
        if old is not None:
            old.hide()
            old.deleteLater()            # not deleted while one of its signals is being handled
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        t = QtWidgets.QLabel(f"<b>{title}</b>")
        lay.addWidget(t)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)
        self.setWidget(w)
        self._w = w
        return lay, form

    def show_for(self, obj):
        """obj: NodeItem / LinkItem / None; shown after the current event (the scene may be rebuilt)."""
        if isinstance(obj, NodeItem):
            ref = ("node", obj.node.id)
        elif isinstance(obj, LinkItem):
            ref = ("link", obj.link)
        else:
            ref = None
        self._pending = ref
        self._timer.start()

    def _show_pending(self):
        self.show_now(self._pending)

    def show_now(self, ref):
        o = self.owner
        d = o.design
        if ref is not None and ref[0] == "node" and ref[1] in d.nodes:
            self._node(d.nodes[ref[1]])
        elif ref is not None and ref[0] == "link" and any(ln is ref[1] for ln in d.links):
            self._link(ref[1])
        else:
            lay, form = self._new("Component " + d.name if d is not o.root else "Structure")
            if d is o.root:
                name = QtWidgets.QLineEdit(d.name)
                name.editingFinished.connect(lambda: o.edit(lambda: setattr(d, "name", name.text().strip() or "structure")))
                form.addRow("name", name)
            n_units = d.count_units() if all(not nd.element_is_component or nd.element in d.components
                                             for nd in d.nodes.values()) else "?"
            form.addRow("nodes", QtWidgets.QLabel(str(len(d.nodes))))
            form.addRow("sub-units", QtWidgets.QLabel(str(n_units)))
            if d is not o.root:
                for which in ("in", "out"):
                    v = d.ports.get(which)
                    txt = f"{d.nodes[v[0]].name}.{v[1]}" if v and v[0] in d.nodes else "— (right-click a port)"
                    form.addRow(f"{which}-port", QtWidgets.QLabel(txt))
            probs = d.problems(component=d is not o.root)
            msg = QtWidgets.QLabel("<br>".join(f"• {p}" for p in probs) if probs else "ready")
            msg.setWordWrap(True)
            msg.setStyleSheet("color: #c8283c;" if probs else "color: #2e8b57;")
            lay.addWidget(msg)
            help_ = QtWidgets.QLabel(
                "Drag sub-units or patterns from the palette onto the canvas, then drag from a port (round: one "
                "point, diamond: distributed points such as a surface) to a port of another node to link them. "
                "Select a node or a link to edit it here.")
            help_.setWordWrap(True)
            help_.setStyleSheet("color: gray;")
            lay.addWidget(help_)
            lay.addStretch(1)

    def _node(self, nd):
        o, d = self.owner, self.owner.design
        lay, form = self._new(("Sub-unit " if nd.is_unit else "Pattern ") + nd.name)
        name = QtWidgets.QLineEdit(nd.name)
        name.editingFinished.connect(lambda: name.text() != nd.name and o.edit(lambda: d.rename(nd.id, name.text().strip())))
        form.addRow("name", name)
        if nd.is_unit:
            kind = QtWidgets.QComboBox()
            kind.addItems(list(UNIT_KINDS))
            kind.setCurrentText(nd.kind)
            kind.currentTextChanged.connect(lambda k: o.edit(lambda: o.change_kind(nd, k)))
            form.addRow("type", kind)
            tag = QtWidgets.QLineEdit(nd.tag)
            tag.setToolTip("sub-units with the same tag share their parameters (beta_<tag>, Rg_<tag>, …)")
            tag.editingFinished.connect(lambda: tag.text() != nd.tag and o.edit(lambda: o.set_tag(nd, tag.text())))
            form.addRow("tag", tag)
            self._options(form, nd, nd.kind)
            params = [f"beta_{nd.tag}"] + [f"{p}_{nd.tag}" for p in REGISTRY[nd.kind].size_params] \
                if nd.kind != "Point" else []
            params += ([f"sigma_{nd.tag}"] if nd.poly else []) + ([f"Rc_{nd.tag}"] if nd.thick else [])
            lab_p = QtWidgets.QLabel(", ".join(params) or "—")
            lab_p.setWordWrap(True)
            form.addRow("parameters", lab_p)
            lab = QtWidgets.QLabel(TIPS.get(nd.kind, ""))
            lab.setWordWrap(True)
            lab.setStyleSheet("color: gray;")
            lay.addWidget(lab)
        else:
            typ = QtWidgets.QComboBox()
            typ.addItems(list(PATTERNS))
            typ.setCurrentText(nd.type)
            typ.currentTextChanged.connect(lambda t: o.edit(lambda: o.change_pattern_type(nd, t)))
            form.addRow("pattern", typ)
            elem = QtWidgets.QComboBox()
            for k in UNIT_KINDS:
                elem.addItem(k, f"unit:{k}")
            for c in sorted(d.components):
                if c != d.name:
                    elem.addItem(f"component {c}", f"comp:{c}")
            i = elem.findData(("comp:" if nd.element_is_component else "unit:") + nd.element)
            elem.setCurrentIndex(max(i, 0))

            def set_elem(_i):
                kind, name = elem.currentData().split(":", 1)
                o.edit(lambda: o.set_element(nd, name, kind == "comp"))
            elem.currentIndexChanged.connect(set_elem)
            form.addRow("element", elem)
            if not nd.element_is_component:
                tag = QtWidgets.QLineEdit(nd.tag)
                tag.editingFinished.connect(lambda: tag.text() != nd.tag and o.edit(lambda: o.set_tag(nd, tag.text())))
                form.addRow("tag", tag)
                refs = REGISTRY[nd.element].references() if nd.element in REGISTRY else ()
                cin = QtWidgets.QComboBox()
                cin.addItems(list(refs))
                cin.setCurrentText(nd.in_ref)
                cin.currentTextChanged.connect(lambda r: o.edit(lambda: setattr(nd, "in_ref", r)))
                form.addRow("attached by" if nd.type == "Array" else "in point", cin)
                if nd.type != "Array":
                    cout = QtWidgets.QComboBox()
                    cout.addItems(list(refs))
                    cout.setCurrentText(nd.out_ref)
                    cout.currentTextChanged.connect(lambda r: o.edit(lambda: setattr(nd, "out_ref", r)))
                    form.addRow("out point", cout)
                self._options(form, nd, nd.element)
            if nd.type in ("Array", "Chain"):
                n = QtWidgets.QSpinBox()
                n.setRange(1, 100000)
                n.setValue(nd.n)
                n.setKeyboardTracking(False)
                n.valueChanged.connect(lambda v: o.edit(lambda: setattr(nd, "n", v)))
                form.addRow("copies", n)
            if nd.type == "Array":
                mode = QtWidgets.QComboBox()
                mode.addItem("independent random points", "independent")
                mode.addItem("all at the same point", "same")
                mode.setCurrentIndex(0 if nd.mode == "independent" else 1)
                mode.setToolTip("at a distributed host point (surface, contour …): each copy at its own random "
                                "point, or all at one; at a specific point always the same")
                mode.currentIndexChanged.connect(lambda _i: o.edit(lambda: setattr(nd, "mode", mode.currentData())))
                form.addRow("where", mode)
                vex = QtWidgets.QCheckBox(f"excluded volume between the copies (vex_{nd.name})")
                vex.setChecked(nd.vex)
                vex.setToolTip("correlation hole between the copies (crowded coronas): their interference terms "
                               "become (1 + k) A..A - k F_copy(q), k = vex/(n - 1), 0 <= vex < 1; the forward "
                               "scattering is unchanged")
                vex.toggled.connect(lambda v: o.edit(lambda: setattr(nd, "vex", v)))
                form.addRow("", vex)
            if nd.type == "Generations":
                f = QtWidgets.QSpinBox()
                f.setRange(1, 20)
                f.setValue(nd.f)
                f.setKeyboardTracking(False)
                f.valueChanged.connect(lambda v: o.edit(lambda: setattr(nd, "f", v)))
                form.addRow("functionality f", f)
                g = QtWidgets.QSpinBox()
                g.setRange(1, 15)
                g.setValue(nd.g)
                g.setKeyboardTracking(False)
                g.valueChanged.connect(lambda v: o.edit(lambda: setattr(nd, "g", v)))
                form.addRow("generations g", g)
                gt = QtWidgets.QCheckBox("own parameters per generation")
                gt.setChecked(nd.gen_tags)
                gt.toggled.connect(lambda v: o.edit(lambda: setattr(nd, "gen_tags", v)))
                form.addRow("", gt)
            try:
                per = d.components[nd.element].count_units() if nd.element_is_component else 1
                tmp = Design(components=d.components)
                tmp.nodes[nd.id] = nd
                cnt = tmp.count_units()
                form.addRow("sub-units", QtWidgets.QLabel(f"{cnt} ({per} per element)"))
            except Exception:  # noqa: BLE001
                pass
            lab = QtWidgets.QLabel(TIPS[nd.type])
            lab.setWordWrap(True)
            lab.setStyleSheet("color: gray;")
            lay.addWidget(lab)
        lay.addStretch(1)

    def _options(self, form, nd, kind):
        """Check boxes: polydisperse (Schulz-Zimm in mass) and thick (cross-section) where the type allows it."""
        from .structure import MASS_SCALING, THICK_KINDS
        o = self.owner
        if kind in MASS_SCALING:
            cb = QtWidgets.QCheckBox(f"polydisperse (sigma_{nd.tag})")
            cb.setChecked(nd.poly)
            cb.setToolTip("Schulz-Zimm distribution of the mass, relative width sigma; sizes scale with the mass, "
                          "beta is proportional to it")
            cb.toggled.connect(lambda v: o.edit(lambda: setattr(nd, "poly", v)))
            form.addRow("", cb)
        if kind in THICK_KINDS:
            cb = QtWidgets.QCheckBox(f"thick (cross-section radius Rc_{nd.tag})")
            cb.setChecked(nd.thick)
            cb.setToolTip("cylindrical cross-section: F x [2 J1(q Rc)/(q Rc)]^2, amplitudes x 2 J1(q Rc)/(q Rc)")
            cb.toggled.connect(lambda v: o.edit(lambda: setattr(nd, "thick", v)))
            form.addRow("", cb)

    def _link(self, ln):
        o, d = self.owner, self.owner.design
        lay, form = self._new("Link")
        for end in (ln.a, ln.b):
            nd = d.nodes[end[0]]
            if d.is_distributed(nd, end[1]):
                box = QtWidgets.QComboBox()
                box.setEditable(True)
                labels = d.labels_on(nd.id, end[1])
                box.addItems(labels)
                box.setCurrentText(end[2])
                box.setToolTip("links with the same label meet at the same random point of this node; "
                               "other labels are independent points")

                def apply(_=None, b=box, e=end):
                    t = b.currentText().strip()
                    if t and t != e[2]:
                        o.edit(lambda: d.set_label(ln, e[0], t))
                box.lineEdit().editingFinished.connect(apply)
                box.activated.connect(apply)
                form.addRow(f"{nd.name}.{end[1]} #", box)
            else:
                form.addRow(f"{nd.name}", QtWidgets.QLabel(end[1]))
        b = QtWidgets.QPushButton("Delete link")
        b.clicked.connect(lambda: o.edit(lambda: d.unlink(ln)))
        lay.addWidget(b)
        lay.addStretch(1)


# ====================================================================== the widget
class DesignerWidget(QtWidgets.QWidget):
    """Palette, canvas and inspector for a :class:`~pysasfit.seb.graph.Design` and its components."""

    changed = QtCore.Signal(object)          # the root design, after every change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = Design("structure")
        self.design = self.root
        self._undo, self._redo = [], []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.addWidget(QtWidgets.QLabel("Editing"))
        self.editing = QtWidgets.QComboBox()
        self.editing.setMinimumContentsLength(14)
        self.editing.currentIndexChanged.connect(self._switch)
        bar.addWidget(self.editing)
        self.tb = QtWidgets.QToolBar()
        self.tb.setIconSize(QtCore.QSize(16, 16))
        self.tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.actions_ = {}
        for key, text, tip, slot, sc in (
                ("newcomp", "+ Component", "new empty component", self.on_new_component, None),
                ("fromsel", "Component from selection", "copy the selected nodes into a new component",
                 self.on_component_from_selection, None),
                ("delcomp", "Delete component", "delete the component being edited (if unused)",
                 self.on_delete_component, None),
                ("sep", None, None, None, None),
                ("undo", "Undo", "undo (Ctrl+Z)", self.undo, QtGui.QKeySequence.StandardKey.Undo),
                ("redo", "Redo", "redo (Ctrl+Y)", self.redo, QtGui.QKeySequence.StandardKey.Redo),
                ("layout", "Layout", "arrange the nodes as a tree", self.on_layout, None),
                ("fit", "Fit", "show the whole design (F); Shift+F: all of it however small",
                 lambda: self.view.fit(), QtGui.QKeySequence("F")),
                ("fitall", None, None, lambda: self.view.fit(min_zoom=0.05), QtGui.QKeySequence("Shift+F")),
                ("delete", "Delete", "delete the selection (Del)", self.delete_selection,
                 QtGui.QKeySequence.StandardKey.Delete),
                ("copy", None, None, self.copy_selection, QtGui.QKeySequence.StandardKey.Copy),
                ("paste", None, None, self.paste, QtGui.QKeySequence.StandardKey.Paste),
                ("dup", None, None, self.duplicate, QtGui.QKeySequence("Ctrl+D")),
                ("all", None, None, self.select_all, QtGui.QKeySequence.StandardKey.SelectAll)):
            if key == "sep":
                self.tb.addSeparator()
                continue
            a = QtGui.QAction(text or key, self)
            if tip:
                a.setToolTip(tip)
            a.triggered.connect(lambda _c=False, s=slot: s())
            if sc is not None:
                a.setShortcut(sc)
                a.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.addAction(a)
            if text:
                self.tb.addAction(a)
            self.actions_[key] = a
        bar.addWidget(self.tb, 1)
        lay.addLayout(bar)
        split = QtWidgets.QSplitter()
        self.palette_ = Palette()
        self.palette_.activated_item.connect(lambda w: self.add_from_palette(w, None))
        self.scene = DesignScene(self)
        self.view = DesignView(self.scene)
        self.view.link_requested.connect(self._on_link)
        self.view.dropped.connect(self.add_from_palette)
        self.view.context.connect(self._on_context)
        self.view.nodes_moved.connect(self._on_moved)
        self.scene.selectionChanged.connect(self._on_selection)
        self.inspector = Inspector(self)
        split.addWidget(self.palette_)
        split.addWidget(self.view)
        split.addWidget(self.inspector)
        split.setStretchFactor(1, 1)
        split.setSizes([170, 600, 250])
        lay.addWidget(split, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.set_design(self.root, fit=True)

    # ------------------------------------------------------------------ design
    def set_design(self, design, fit=True, keep_undo=False):
        """Show a new root design (template, file, session)."""
        if not keep_undo:
            self._undo, self._redo = [], []
        self.root = design
        self.design = design
        self._refresh_editing()
        self.refresh()
        if fit:
            QtCore.QTimer.singleShot(0, self.view.fit)

    def _refresh_editing(self):
        self.editing.blockSignals(True)
        self.editing.clear()
        self.editing.addItem("structure", None)
        for c in sorted(self.root.components):
            self.editing.addItem(f"component {c}", c)
        i = self.editing.findData(None if self.design is self.root else self.design.name)
        self.editing.setCurrentIndex(max(i, 0))
        self.editing.blockSignals(False)
        self.palette_.set_components(sorted(self.root.components))
        self.actions_["delcomp"].setEnabled(self.design is not self.root)

    def _switch(self, i):
        name = self.editing.itemData(i)
        self.design = self.root if name is None else self.root.components[name]
        self.actions_["delcomp"].setEnabled(self.design is not self.root)
        self.refresh()
        QtCore.QTimer.singleShot(0, self.view.fit)

    def refresh(self):
        self.scene.show_design(self.design)
        self.inspector.show_for(self._selected_one())
        probs = self.root.problems()
        self.status.setText("<span style='color:#c8283c'>" + "; ".join(probs) + "</span>" if probs else
                            f"{self.root.count_units()} sub-units")
        self.actions_["undo"].setEnabled(bool(self._undo))
        self.actions_["redo"].setEnabled(bool(self._redo))

    def _selected_one(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, (NodeItem, LinkItem))]
        return sel[0] if len(sel) == 1 else None

    def _on_selection(self):
        try:
            self.inspector.show_for(self._selected_one())
        except RuntimeError:       # items deleted during a rebuild
            pass

    # ------------------------------------------------------------------ editing with undo
    def _snapshot(self):
        return json.dumps({"root": self.root.to_dict(), "editing": None if self.design is self.root
                           else self.design.name})

    def _restore(self, snap):
        st = json.loads(snap)
        self.root = Design.from_dict(st["root"])
        ed = st.get("editing")
        self.design = self.root.components.get(ed, self.root) if ed else self.root
        self._refresh_editing()
        self.refresh()
        self.changed.emit(self.root)

    def checkpoint(self):
        self._undo.append(self._snapshot())
        del self._undo[:-200]
        self._redo.clear()

    def edit(self, fn):
        """Apply fn() as one undoable change; errors are shown, the design stays as it was."""
        snap = self._snapshot()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- show, don't crash
            st = json.loads(snap)
            self.root = Design.from_dict(st["root"])
            ed = st.get("editing")
            self.design = self.root.components.get(ed, self.root) if ed else self.root
            self.refresh()
            self.status.setText(f"<span style='color:#c8283c'>{exc}</span>")
            return False
        self._undo.append(snap)
        del self._undo[:-200]
        self._redo.clear()
        self._refresh_editing()
        self.refresh()
        self.changed.emit(self.root)
        return True

    def undo(self):
        if self._undo:
            self._redo.append(self._snapshot())
            self._restore(self._undo.pop())

    def redo(self):
        if self._redo:
            self._undo.append(self._snapshot())
            self._restore(self._redo.pop())

    def _on_moved(self):
        # positions are already in the nodes; one undo step for the move
        self._undo.append(self._snapshot())
        self._redo.clear()
        self.refresh_undo_buttons()

    def refresh_undo_buttons(self):
        self.actions_["undo"].setEnabled(bool(self._undo))
        self.actions_["redo"].setEnabled(bool(self._redo))

    # ------------------------------------------------------------------ operations
    def add_from_palette(self, what, pos):
        if pos is None:
            pos = self.view.mapToScene(self.view.viewport().rect().center()) - QtCore.QPointF(80, 30)
        typ, name = what.split(":", 1)
        d = self.design
        out = {}

        def do():
            if typ == "unit":
                out["n"] = d.add_unit(name, x=pos.x(), y=pos.y())
            elif typ == "pattern":
                out["n"] = d.add_pattern(name, "GaussianPolymer", x=pos.x(), y=pos.y())
            else:
                out["n"] = d.add_pattern("Array", name, component=True, x=pos.x(), y=pos.y())
        if self.edit(do) and "n" in out:
            it = self.scene.node_items.get(out["n"].id)
            if it is not None:
                self.scene.clearSelection()
                it.setSelected(True)

    def _on_link(self, pa, pb):
        na, nb = pa.node_item.node, pb.node_item.node
        ok, why = self.design.can_link(na.id, pa.port, nb.id, pb.port)
        if not ok:
            self.status.setText(f"<span style='color:#c8283c'>{why}</span>")
            return
        self.edit(lambda: self.design.link(na.id, pa.port, nb.id, pb.port))

    def delete_selection(self):
        items = self.scene.selectedItems()
        nodes = [i.node.id for i in items if isinstance(i, NodeItem)]
        links = [i.link for i in items if isinstance(i, LinkItem)]
        if not nodes and not links:
            return

        def do():
            for ln in links:
                self.design.unlink(ln)
            for nid in nodes:
                self.design.remove_node(nid)
        self.edit(do)

    def select_all(self):
        for it in self.scene.items():
            if isinstance(it, (NodeItem, LinkItem)):
                it.setSelected(True)

    def _selection_dict(self):
        ids = {i.node.id for i in self.scene.selectedItems() if isinstance(i, NodeItem)}
        if not ids:
            return None
        d = self.design
        return {"nodes": [d.nodes[i].to_dict() for i in ids],
                "links": [ln.to_dict() for ln in d.links if ln.a[0] in ids and ln.b[0] in ids]}

    def copy_selection(self):
        sel = self._selection_dict()
        if sel is None:
            return
        # plain text on the clipboard: a QMimeData made in Python and owned by the clipboard crashes the
        # interpreter at exit (PySide6 deletes it twice)
        txt = json.dumps({"format": MIME_NODES, **sel})
        QtWidgets.QApplication.clipboard().setText(txt)
        self._clip = txt

    def paste(self, txt=None):
        if txt is None:
            txt = QtWidgets.QApplication.clipboard().text()
            if f'"{MIME_NODES}"' not in (txt or "")[:80]:
                txt = getattr(self, "_clip", None)
        if not txt:
            return
        try:
            sel = json.loads(txt)
        except ValueError:
            return
        d = self.design
        new_ids = {}

        def do():
            from .graph import Link, Node
            for nd in sel["nodes"]:
                n = Node.from_dict(nd)
                old = n.id
                n.id = d._next
                d._next += 1
                n.name = d.unique_name(re.sub(r"\d+$", "", n.name) or "u")
                n.x += 40
                n.y += 40
                d.nodes[n.id] = n
                new_ids[old] = n.id
            for ln in sel["links"]:
                a, b = list(ln["a"]), list(ln["b"])
                a[0], b[0] = new_ids[a[0]], new_ids[b[0]]
                d.links.append(Link(a, b))
        if self.edit(do):
            self.scene.clearSelection()
            for nid in new_ids.values():
                it = self.scene.node_items.get(nid)
                if it is not None:
                    it.setSelected(True)

    def duplicate(self):
        sel = self._selection_dict()
        if sel is not None:
            self.paste(json.dumps(sel))

    def on_layout(self):
        self.edit(self.design.auto_layout)
        QtCore.QTimer.singleShot(0, self.view.fit)

    # node property changes (called through edit)
    def change_kind(self, nd, kind):
        nd.kind = kind
        refs = REGISTRY[kind].references()
        d = self.design
        d.links = [ln for ln in d.links if not ((ln.a[0] == nd.id and ln.a[1] not in refs)
                                                 or (ln.b[0] == nd.id and ln.b[1] not in refs))]
        d.ports = {k: v for k, v in d.ports.items() if not (v[0] == nd.id and v[1] not in refs)}

    def change_pattern_type(self, nd, t):
        from .graph import PATTERN_PORTS
        nd.type = t
        d = self.design
        ports = PATTERN_PORTS[t]
        keep = []
        for ln in d.links:
            e = ln.end_of(nd.id) if nd.id in (ln.a[0], ln.b[0]) else None
            if e is None:
                keep.append(ln)
                continue
            if e[1] in ports:
                keep.append(ln)
            elif ports:
                # re-attach to the pattern's first port
                if ln.a[0] == nd.id:
                    ln.a = (nd.id, ports[0], "")
                else:
                    ln.b = (nd.id, ports[0], "")
                keep.append(ln)
        d.links = keep
        if t == "Array":
            mine = [ln for ln in d.links if nd.id in (ln.a[0], ln.b[0])]
            for ln in mine[1:]:
                d.unlink(ln)
        if not nd.element_is_component and nd.element in REGISTRY:
            refs = REGISTRY[nd.element].references()
            nd.out_ref = nd.out_ref if nd.out_ref in refs else refs[-1]

    def set_element(self, nd, element, is_comp):
        nd.element, nd.element_is_component = element, is_comp
        if not is_comp:
            refs = REGISTRY[element].references()
            nd.in_ref = nd.in_ref if nd.in_ref in refs else ("end1" if "end1" in refs else refs[0])
            nd.out_ref = nd.out_ref if nd.out_ref in refs else ("end2" if "end2" in refs else refs[-1])
            nd.tag = nd.tag or short_name(element)

    def set_tag(self, nd, tag):
        tag = tag.strip()
        if not re.match(r"^[A-Za-z0-9_]+$", tag):
            raise ValueError("tags contain only letters, digits and _")
        nd.tag = tag

    # ------------------------------------------------------------------ components
    def on_new_component(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New component", "name of the component")
        if ok and name:
            self._make_component(lambda: self.root.new_component(name.strip()))

    def on_component_from_selection(self):
        ids = [i.node.id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not ids:
            self.status.setText("select the nodes of the new component first")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Component from selection", "name of the component")
        if ok and name:
            self._make_component(lambda: self.design.component_from(ids, name.strip()))

    def _make_component(self, fn):
        out = {}

        def do():
            out["c"] = fn()
        if self.edit(do):
            self.design = out["c"]
            self._refresh_editing()
            self.refresh()
            self.status.setText(f"component '{out['c'].name}': right-click a port of a sub-unit to make it the "
                                "in-port (and the out-port for chains and generations)")

    def on_delete_component(self):
        c = self.design
        if c is self.root:
            return
        users = [nd.name for nd in self.root.nodes.values() if not nd.is_unit and nd.element_is_component
                 and nd.element == c.name]
        users += [f"{o.name}.{nd.name}" for o in self.root.components.values() for nd in o.nodes.values()
                  if not nd.is_unit and nd.element_is_component and nd.element == c.name]
        if users:
            self.status.setText(f"<span style='color:#c8283c'>component '{c.name}' is used by "
                                f"{', '.join(users)}</span>")
            return

        def do():
            del self.root.components[c.name]
            self.design = self.root
        self.edit(do)

    def _on_context(self, item, gpos):
        m = QtWidgets.QMenu(self)
        d = self.design
        if isinstance(item, PortItem) and d is not self.root and item.node_item.node.is_unit:
            nd, port = item.node_item.node, item.port
            m.addAction(f"Make {nd.name}.{port} the in-port",
                        lambda: self.edit(lambda: d.set_port("in", nd.id, port)))
            m.addAction(f"Make {nd.name}.{port} the out-port",
                        lambda: self.edit(lambda: d.set_port("out", nd.id, port)))
        elif isinstance(item, PortItem):
            m.addAction("Drag from a port to another port to link them").setEnabled(False)
        if isinstance(item, (NodeItem, PortItem)):
            ni = item if isinstance(item, NodeItem) else item.node_item
            if not ni.isSelected():
                self.scene.clearSelection()
                ni.setSelected(True)
            m.addAction("Duplicate (Ctrl+D)", self.duplicate)
            m.addAction("Delete (Del)", self.delete_selection)
            sel_nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
            if sel_nodes:
                m.addAction("Component from selection…", self.on_component_from_selection)
        elif isinstance(item, LinkItem):
            item.setSelected(True)
            m.addAction("Delete link", lambda: self.edit(lambda: d.unlink(item.link)))
        else:
            add = m.addMenu("Add")
            pos = self.view.mapToScene(self.view.mapFromGlobal(gpos))
            for k in UNIT_KINDS:
                add.addAction(k, lambda kk=k: self.add_from_palette(f"unit:{kk}", pos))
            add.addSeparator()
            for k in PATTERNS:
                add.addAction(k, lambda kk=k: self.add_from_palette(f"pattern:{kk}", pos))
            m.addAction("Layout", self.on_layout)
            m.addAction("Fit (F)", self.view.fit)
        m.exec(gpos)

