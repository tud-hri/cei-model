from PyQt5 import QtWidgets, QtCore, QtGui


class BoxPlotItem(QtWidgets.QGraphicsItemGroup):
    def __init__(self, sigma, color=QtCore.Qt.white, parent=None):
        super().__init__(parent)

        inter_quartile_range = 1.34896 * sigma
        self.rectangle = QtWidgets.QGraphicsRectItem(-inter_quartile_range/2., -0.5, inter_quartile_range, 1.)
        pen = QtGui.QPen()
        pen.setColor(color)
        pen.setWidthF(0.1)
        self.rectangle.setPen(pen)
        self.rectangle.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        self.line = QtWidgets.QGraphicsLineItem(-3 * inter_quartile_range, 0.0, 3 * inter_quartile_range, 0.0)
        self.line.setPen(pen)

        self.addToGroup(self.rectangle)
        self.addToGroup(self.line)

    def rescale(self, sigma):
        inter_quartile_range = 1.34896 * sigma
        self.rectangle.setRect(-inter_quartile_range/2., -0.5, inter_quartile_range, 1.)
        self.line.setLine(-3 * inter_quartile_range, 0.0, 3 * inter_quartile_range, 0.0)
