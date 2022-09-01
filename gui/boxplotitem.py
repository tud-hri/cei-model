"""
Copyright 2022, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of the CEI-model repository.

The CEI-model repository is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The CEI-model repository is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the CEI-model repository. If not, see <https://www.gnu.org/licenses/>.
"""
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
