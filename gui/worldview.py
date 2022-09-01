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
import os
import random

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from gui.boxplotitem import BoxPlotItem
from trackobjects.trackside import TrackSide


class WorldView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_gui = None
        self.road_graphics = None
        self.track = None

        self.scene = QtWidgets.QGraphicsScene()
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setScene(self.scene)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(14, 150, 22)))

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self.horizontal_scale_factor = 0.0
        self.vertical_scale_factor = 0.0

        # Scaled size zoomRect
        self.max_zoom_size = 0.0
        self.min_zoom_size = 0.0
        self.zoom_level = 0.0
        self.zoom_center = None

        # intialize overlay
        self._overlay_message = 'Not started'
        self._draw_overlay = True

        self.controllable_objects = []
        self.graphics_objects = []

        # initialize plan and believe point lists
        self.plan_graphics_objects = []
        self.belief_graphics_objects = []

    def initialize(self, track, main_gui, number_of_belief_points, show_run_up=False):
        self.main_gui = main_gui
        self.road_graphics = QtWidgets.QGraphicsItemGroup()
        self.track = track

        for way_point_set in [track.get_way_points(TrackSide.LEFT, show_run_up), track.get_way_points(TrackSide.RIGHT, show_run_up)]:
            road_path = QtWidgets.QGraphicsPathItem()
            road_painter = QtGui.QPainterPath()
            pen = QtGui.QPen()
            pen.setWidthF(track.track_width)
            pen.setColor(QtGui.QColor(50, 50, 50))
            road_path.setPen(pen)

            center_line_path = QtWidgets.QGraphicsPathItem()
            center_line_painter = QtGui.QPainterPath()
            pen = QtGui.QPen()
            pen.setWidthF(0.02)
            pen.setColor(QtGui.QColor(255, 255, 255))
            center_line_path.setPen(pen)

            road_painter.moveTo(way_point_set[0][0], -way_point_set[0][1])
            center_line_painter.moveTo(way_point_set[0][0], -way_point_set[0][1])

            for way_point in way_point_set[1:]:
                road_painter.lineTo(way_point[0], -way_point[1])
                center_line_painter.lineTo(way_point[0], -way_point[1])

            road_path.setPath(road_painter)
            center_line_path.setPath(center_line_painter)
            center_line_path.setZValue(1.0)

            self.road_graphics.addToGroup(road_path)
            self.road_graphics.addToGroup(center_line_path)

        self.road_graphics.setZValue(1.0)
        self.scene.addItem(self.road_graphics)
        padding_rect_size = self.road_graphics.sceneBoundingRect().size() * 4.0
        padding_rect_top_left_x = self.road_graphics.sceneBoundingRect().center().x() - padding_rect_size.width() / 2
        padding_rect_top_left_y = -self.road_graphics.sceneBoundingRect().center().y() - padding_rect_size.height() / 2
        scroll_padding_rect = QtWidgets.QGraphicsRectItem(padding_rect_top_left_x, padding_rect_top_left_y, padding_rect_size.width(),
                                                          padding_rect_size.height())
        scroll_padding_rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scene.addItem(scroll_padding_rect)

        # initialize belief and plan graphics
        for belief_index in range(number_of_belief_points):
            plan_pen = QtGui.QPen()
            plan_pen.setColor(QtCore.Qt.red)
            plan_pen.setWidthF(0.1)
            plan_graphics = QtWidgets.QGraphicsEllipseItem(-0.5, -0.5, 1.0, 1.0)
            plan_graphics.setPen(plan_pen)
            plan_graphics.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))

            plan_graphics.setVisible(False)
            plan_graphics.setZValue(100.)
            self.plan_graphics_objects.append(plan_graphics)

            belief_graphics = BoxPlotItem(1.0)
            belief_graphics.setVisible(False)
            belief_graphics.setZValue(100.)
            self.belief_graphics_objects.append(belief_graphics)

            self.scene.addItem(plan_graphics)
            self.scene.addItem(belief_graphics)

        # Scaled size zoomRect
        self.max_zoom_size = self.road_graphics.sceneBoundingRect().size() * 1
        self.min_zoom_size = self.road_graphics.sceneBoundingRect().size() * 0.01
        self.zoom_level = 0.0
        self.zoom_center = self.road_graphics.sceneBoundingRect().center()
        self.update_zoom()

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        radius = 0.1
        graphics = QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)

        graphics.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        graphics.setBrush(color)

        graphics.setZValue(2.0)
        self.scene.addItem(graphics)
        self.controllable_objects.append(controllable_object)
        self.graphics_objects.append(graphics)
        self.update_all_graphics_positions()

    def add_controllable_car(self, controllable_object, vehicle_length, vehicle_width, color='red'):
        """
        adds a controllable car object to the scene, color should be one of {blue, green, purple, red, white, yellow}
        """
        file_name = color.lower().strip() + '_car.png'
        file_path = 'images/' + file_name

        if not os.path.isfile(file_path):
            raise RuntimeError('Car graphics for color: ' + color + ' can not be found. Looking for the file: ' + file_path +
                               '. Please specify a different color.')

        vehicle_pixmap = QtGui.QPixmap(file_path)
        graphics_object = QtWidgets.QGraphicsPixmapItem(vehicle_pixmap)
        graphics_object.setOffset(-vehicle_pixmap.width() / 2, -vehicle_pixmap.height() / 2)
        graphics_object.setTransformationMode(QtCore.Qt.SmoothTransformation)
        graphics_object.setZValue(2.0)

        self.horizontal_scale_factor = vehicle_length / vehicle_pixmap.width()
        self.vertical_scale_factor = vehicle_width * 1.2 / vehicle_pixmap.height()
        # the mirrors on the vehicle graphics account for 20% of the width but are not included in the vehicle_width, hence the 1.2 factor

        self.scene.addItem(graphics_object)
        self.controllable_objects.append(controllable_object)
        self.graphics_objects.append(graphics_object)
        self.update_all_graphics_positions()

    def set_plan_and_belief_visible(self, boolean):
        for plan_graphics in self.plan_graphics_objects:
            plan_graphics.setVisible(boolean)

        for belief_graphics in self.belief_graphics_objects:
            belief_graphics.setVisible(boolean)

    def update_plan_and_belief_graphics(self, position_plan, belief):
        number_of_points = len(self.belief_graphics_objects)
        slices = int(len(position_plan) / number_of_points)

        plan = np.array(position_plan)[0::slices]
        belief = np.array(belief)[0::slices]

        for plan_graphics, plan_point in zip(self.plan_graphics_objects, plan):
            position = self.track.traveled_distance_to_coordinates(plan_point, TrackSide.LEFT)
            plan_graphics.setPos(position[0], -position[1])

        for belief_graphics, belief_point in zip(self.belief_graphics_objects, belief):
            belief_graphics.rescale(belief_point[1])
            position = self.track.traveled_distance_to_coordinates(belief_point[0], TrackSide.RIGHT)
            angle = self.track.get_heading(position)

            belief_graphics.setPos(position[0], -position[1])
            belief_graphics.setRotation(-np.degrees(angle))

    def update_all_graphics_positions(self):
        for controllable_object, graphics_object in zip(self.controllable_objects, self.graphics_objects):
            transform = QtGui.QTransform()
            transform.rotate(-np.degrees(controllable_object.heading))
            transform.scale(self.horizontal_scale_factor, self.vertical_scale_factor)

            graphics_object.setTransform(transform)

            graphics_object.setPos(controllable_object.position[0], -controllable_object.position[1])

    def update_zoom(self):
        # Compute scale factors (in x- and y-direction)
        zoom = (1.0 - self.zoom_level) ** 2
        scale1 = zoom + (self.min_zoom_size.width() / self.max_zoom_size.width()) * (1.0 - zoom)
        scale2 = zoom + (self.min_zoom_size.height() / self.max_zoom_size.height()) * (1.0 - zoom)

        # Scaled size zoomRect
        scaled_w = self.max_zoom_size.width() * scale1
        scaled_h = self.max_zoom_size.height() * scale2

        # Set zoomRect
        view_zoom_rect = QtCore.QRectF(self.zoom_center.x() - scaled_w / 2, self.zoom_center.y() - scaled_h / 2, scaled_w, scaled_h)

        # Set view (including padding)
        self.fitInView(view_zoom_rect, QtCore.Qt.KeepAspectRatio)

    def set_overlay_message(self, message):
        self._overlay_message = message

    def draw_overlay(self, bool):
        self._draw_overlay = bool
        self.scene.update()

    def drawForeground(self, painter, rect):
        if self._draw_overlay:

            painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            painter.setPen(QtGui.QPen())
            painter.setOpacity(0.3)

            # create rectangle with 20% margin around the edges for smooth panning
            corner = self.mapToScene(QtCore.QPoint(-0.2 * self.width(), -0.2 * self.height()))
            painter.drawRect(corner.x(), corner.y(), 1.4 * self.width(), 1.4 * self.height())

            painter.setOpacity(1.0)
            font = QtGui.QFont()
            font.setPointSize(3)
            font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 130.)

            painter_path = QtGui.QPainterPath()
            painter.setBrush(QtCore.Qt.white)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

            text_width = QtGui.QFontMetrics(font).horizontalAdvance(self._overlay_message)
            painter_path.addText(rect.center().x() - text_width / 2, rect.center().y(), font, self._overlay_message)

            painter.drawPath(painter_path)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_zoom()

    def wheelEvent(self, event):
        direction = np.sign(event.angleDelta().y())
        self.zoom_level = max(min(self.zoom_level + direction * 0.1, 1.0), 0.0)
        self.update_zoom()

    def enterEvent(self, e):
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        super().enterEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:  # Drag scene
            self.zoom_center = self.mapToScene(self.rect().center())
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.buttons() == QtCore.Qt.MiddleButton:  # Drag scene
            self.main_gui.statusBar.showMessage('position of mouse: %0.1f, %0.1f  -  position of point mass: %0.1f, %0.1f  ' % (
                self.mapToScene(e.pos()).x(), -self.mapToScene(e.pos()).y(), self.controllable_objects[0].position[0],
                -self.controllable_objects[0].position[1]))
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        self.update_zoom()
