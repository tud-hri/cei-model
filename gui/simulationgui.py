import datetime
import os
import pickle

import cv2
import numpy as np
import pyqtgraph
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import stats

from agents import CEIAgent
from trackobjects.trackside import TrackSide
from .simulation_gui_ui import Ui_SimpleMerging


class SimulationGui(QtWidgets.QMainWindow):
    def __init__(self, track, show_plotting_pane=True, in_replay_mode=False, number_of_belief_points=0, surroundings=None, enable_print_style_plots=False,
                 parent=None):
        super().__init__(parent)

        if enable_print_style_plots:
            pyqtgraph.setConfigOption('background', 'w')
            pyqtgraph.setConfigOption('foreground', 'k')

        self.ui = Ui_SimpleMerging()
        self.ui.setupUi(self)

        self.ui.world_view.initialize(track, self, number_of_belief_points)
        if surroundings:
            self.ui.world_view.scene.addItem(surroundings.get_graphics_objects())

        self.ui.leftSpeedDial.initialize(min_velocity=0.0, max_velocity=20.)
        self.ui.rightSpeedDial.initialize(min_velocity=0.0, max_velocity=20.)

        self.track = track
        self.show_plotting_pane = show_plotting_pane
        self.in_replay_mode = in_replay_mode

        self.ui.play_button.clicked.connect(self.toggle_play)
        self.ui.previous_button.clicked.connect(self.previous_frame)
        self.ui.next_button.clicked.connect(self.next_frame)
        self.ui.next_button.setEnabled(False)
        self.ui.previous_button.setEnabled(False)
        self.ui.timeSlider.setEnabled(in_replay_mode)

        self.ui.timeSlider.sliderReleased.connect(self._set_time)

        self.is_expanded = True
        self.ui.expandPushButton.clicked.connect(self._expand_window)

        if not self.show_plotting_pane:
            self._expand_window()
            self.ui.expandPushButton.setEnabled(False)
            self.ui.expandPushButton.setVisible(False)

        self.ui.worldViewPlotCheckBox.stateChanged.connect(self._toggle_world_view_plots)

        self._time_indicator_lines = []
        self._distance_indicator_lines = []
        self.average_travelled_distance_trace = None

        self.video_writer = None
        self.is_recording = False
        self.path_to_video_file = ''

        self.ui.actionEnable_recording.triggered.connect(self._enable_recording)

        self.show()

    def _expand_window(self):
        if self.is_expanded:
            self.ui.tabWidget.setVisible(False)
            self.ui.expandPushButton.setText('>')
            self.resize(650, 650)
        else:
            self.ui.tabWidget.setVisible(True)
            self.ui.expandPushButton.setText('<')
            self.resize(1300, 650)

        self.is_expanded = not self.is_expanded

    def register_sim_master(self, sim_master):
        self.sim_master = sim_master

    def toggle_play(self):
        if self.sim_master and not self.sim_master.main_timer.isActive():
            self.sim_master.start()
            if self.in_replay_mode:
                self.ui.play_button.setText('Pause')
                self.ui.next_button.setEnabled(False)
                self.ui.previous_button.setEnabled(False)
            else:
                self.ui.play_button.setEnabled(False)
        elif self.sim_master:
            self.sim_master.pause()
            self.ui.play_button.setText('Play')
            if self.in_replay_mode:
                self.ui.next_button.setEnabled(True)
                self.ui.previous_button.setEnabled(True)

    def reset(self):
        self.ui.play_button.setText('Play')
        self.ui.play_button.setEnabled(True)
        if self.in_replay_mode:
            self.ui.next_button.setEnabled(True)
            self.ui.previous_button.setEnabled(True)

    def next_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step()

    def previous_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step(reverse=True)

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        self.ui.world_view.add_controllable_dot(controllable_object, color)

    def add_controllable_car(self, controllable_object, vehicle_length, vehicle_width, side_for_dial, color='red'):
        self.ui.world_view.add_controllable_car(controllable_object, vehicle_length, vehicle_width, color)

        if side_for_dial is TrackSide.LEFT:
            self.ui.leftSpeedDial.set_velocity(controllable_object.velocity)
        elif side_for_dial is TrackSide.RIGHT:
            self.ui.rightSpeedDial.set_velocity(controllable_object.velocity)

    def update_all_graphics(self, left_velocity, right_velocity):
        self.ui.world_view.update_all_graphics_positions()

        self.ui.leftSpeedDial.set_velocity(left_velocity)
        self.ui.rightSpeedDial.set_velocity(right_velocity)

    def update_time_label(self, time):
        self.ui.statusbar.showMessage('time: %0.2f s' % time)
        if self.in_replay_mode:
            time_promille = int(self.sim_master.time_index * 1000 / self.sim_master.maxtime_index)
            self.ui.timeSlider.setValue(time_promille)

            if self.ui.tabWidget.currentIndex() == 1:
                self._update_trace_plots()

    def show_overlay(self, message=None):
        if message:
            self.ui.world_view.set_overlay_message(message)
            self.ui.world_view.draw_overlay(True)
        else:
            self.ui.world_view.draw_overlay(False)

    def _set_time(self):
        time_promille = self.ui.timeSlider.value()
        self.sim_master.set_time(time_promille=time_promille)

    @staticmethod
    def _add_padding_to_plot_widget(plot_widget, padding=0.1):
        """
        zooms out the view of a plot widget to show 'padding' around the contents of a PlotWidget
        :param plot_widget: The widget to add padding to
        :param padding: the percentage of padding expressed between 0.0 and 1.0
        :return:
        """

        width = plot_widget.sceneRect().width() * (1. + padding)
        height = plot_widget.sceneRect().height() * (1. + padding)
        center = plot_widget.sceneRect().center()
        zoom_rect = QtCore.QRectF(center.x() - width / 2., center.y() - height / 2., width, height)

        plot_widget.fitInView(zoom_rect)

    def initialize_plots(self, data_dict, time, left_color, right_color, plot_live_plot=True):
        self._initialize_trace_plots(data_dict, time, left_color, right_color)

        if plot_live_plot:
            belief = data_dict['beliefs'][TrackSide.LEFT][0][0:-1]
            belief_time_stamps = data_dict['belief_time_stamps'][TrackSide.LEFT][0][0:-1]
            position_plan = data_dict['position_plans'][TrackSide.LEFT][0]
            action_plan = data_dict['action_plans'][TrackSide.LEFT][0]
            perceived_risk = data_dict['perceived_risks'][TrackSide.LEFT][0]
            left_risk_bounds = data_dict['risk_bounds'][TrackSide.LEFT]
            did_replanning = data_dict['is_replanning'][TrackSide.LEFT][0]
            belief_point_contributing_to_risk = data_dict['belief_point_contributing_to_risk'][TrackSide.LEFT][0]

            self._update_live_plots(belief, belief_time_stamps, position_plan, action_plan, perceived_risk, left_risk_bounds, did_replanning,
                                    belief_point_contributing_to_risk)

        self.ui.graphicsView.setTitle('Belief and Plan')
        self.ui.accelerationGraphicsView.setTitle('Acceleration [m/s<sup>2</sup>]')
        self.ui.velocityGraphicsView.setTitle('Velocity [m/s]')
        self.ui.sepperationGraphicsView.setTitle('Headway [m]')
        self.ui.riskGraphicsView.setTitle('Perceived Risk [-]')
        self.ui.riskGraphicsView.setLabel('bottom', 'time [s]')

        self.ui.graphicsView.setYRange(0., 1.)
        self.ui.graphicsView.setXRange(0., self.track.total_distance)

        self._add_padding_to_plot_widget(self.ui.graphicsView)
        self._add_padding_to_plot_widget(self.ui.accelerationGraphicsView)
        self._add_padding_to_plot_widget(self.ui.velocityGraphicsView)
        self._add_padding_to_plot_widget(self.ui.sepperationGraphicsView)
        self._add_padding_to_plot_widget(self.ui.riskGraphicsView)

    def _toggle_world_view_plots(self):
        self.ui.world_view.set_plan_and_belief_visible(self.ui.worldViewPlotCheckBox.isChecked())

    def update_plots(self, belief, belief_time_stamps, position_plan, action_plan, perceived_risk, risk_bounds, did_replanning,
                     belief_point_contributing_to_risk):

        if self.show_plotting_pane and self.ui.tabWidget.currentIndex() == 0:
            self._update_live_plots(belief, belief_time_stamps, position_plan, action_plan, perceived_risk, risk_bounds, did_replanning,
                                    belief_point_contributing_to_risk)

        if self.ui.worldViewPlotCheckBox.isChecked():
            belief_times_from_now = np.array(belief_time_stamps) - (self.sim_master.t / 1000.)
            position_indices = ((belief_times_from_now / (self.sim_master.dt / 1000.)) - 1).astype(int)

            position_plan_corresponding_to_belief = position_plan[position_indices]

            self.ui.world_view.update_plan_and_belief_graphics(position_plan_corresponding_to_belief, belief)

    def _initialize_trace_plots(self, data_dict, time, left_color, right_color):

        if data_dict['agent_types'][TrackSide.LEFT] == CEIAgent:
            left_risk_bounds = data_dict['risk_bounds'][TrackSide.LEFT]
            right_risk_bounds = data_dict['risk_bounds'][TrackSide.RIGHT]
            if not right_risk_bounds:
                right_risk_bounds = (0., 0.)

            left_risk_trace = data_dict['perceived_risks'][TrackSide.LEFT]
            right_risk_trace = data_dict['perceived_risks'][TrackSide.RIGHT]

            if not right_risk_trace:
                right_risk_trace = [0.] * len(left_risk_trace)
        else:
            left_risk_bounds = None
            right_risk_bounds = None
            left_risk_trace = None
            right_risk_trace = None

        left_acceleration_trace = data_dict['net_accelerations'][TrackSide.LEFT]
        right_acceleration_trace = data_dict['net_accelerations'][TrackSide.RIGHT]

        cooperation_trace = -1 * np.array(left_acceleration_trace) * np.array(right_acceleration_trace)

        if not right_acceleration_trace:
            right_acceleration_trace = [0.] * len(left_acceleration_trace)

        left_velocity_trace = data_dict['velocities'][TrackSide.LEFT]
        right_velocity_trace = data_dict['velocities'][TrackSide.RIGHT]
        position_left = data_dict['travelled_distance'][TrackSide.LEFT]
        position_right = data_dict['travelled_distance'][TrackSide.RIGHT]

        headway_trace = np.array(position_left) - np.array(position_right)
        self.average_travelled_distance_trace = (np.array(position_left) + np.array(position_right)) / 2.

        left_pen = pyqtgraph.mkPen(left_color, width=1.5)
        right_pen = pyqtgraph.mkPen(right_color, width=1.5)
        white_pen = pyqtgraph.mkPen('w', width=1.5)
        left_brush = pyqtgraph.mkBrush(left_color)
        right_brush = pyqtgraph.mkBrush(right_color)
        white_brush = pyqtgraph.mkBrush('w')

        if left_risk_trace is None and right_risk_trace is None:
            self.ui.riskGraphicsView.setVisible(False)

        for x, y, view, pen, brush, color, bounds in [(time, left_acceleration_trace, self.ui.accelerationGraphicsView, left_pen, left_brush, left_color, None),
                                                      (time, right_acceleration_trace, self.ui.accelerationGraphicsView, right_pen, right_brush, right_color,
                                                       None),
                                                      (time, left_velocity_trace, self.ui.velocityGraphicsView, left_pen, left_brush, left_color, None),
                                                      (time, right_velocity_trace, self.ui.velocityGraphicsView, right_pen, right_brush, right_color, None),
                                                      (time, cooperation_trace, self.ui.coopGraphicsView, white_pen, white_brush, 'w', None),
                                                      (self.average_travelled_distance_trace, headway_trace, self.ui.sepperationGraphicsView, white_pen,
                                                       white_brush, 'w', None),
                                                      (time, left_risk_trace, self.ui.riskGraphicsView, left_pen, left_brush, left_color, left_risk_bounds),
                                                      (
                                                      time, right_risk_trace, self.ui.riskGraphicsView, right_pen, left_brush, right_color, right_risk_bounds)]:
            if y is not None:
                view.plot(x, y, pen=pen)
            if bounds is not None:
                upper_risk_bound = pyqtgraph.InfiniteLine(pos=bounds[1], pen=pyqtgraph.mkPen(color, width=1.5, style=QtCore.Qt.DashLine), angle=0.)
                lower_risk_bound = pyqtgraph.InfiniteLine(pos=bounds[0], pen=pyqtgraph.mkPen(color, width=1.5, style=QtCore.Qt.DashLine), angle=0.)
                view.addItem(upper_risk_bound)
                view.addItem(lower_risk_bound)

        for plot_view in [self.ui.accelerationGraphicsView, self.ui.velocityGraphicsView, self.ui.coopGraphicsView, self.ui.riskGraphicsView]:
            time_indicator_line = pyqtgraph.InfiniteLine(pos=0., pen=pyqtgraph.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
            self._time_indicator_lines.append(time_indicator_line)
            plot_view.addItem(time_indicator_line)

        distance_indicator_line = pyqtgraph.InfiniteLine(pos=self.average_travelled_distance_trace[0],
                                                         pen=pyqtgraph.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
        self._distance_indicator_lines.append(distance_indicator_line)
        self.ui.sepperationGraphicsView.addItem(distance_indicator_line)

        zero_line = pyqtgraph.InfiniteLine(pos=0., angle=0., pen=pyqtgraph.mkPen('grey', width=1., style=QtCore.Qt.DashLine))
        self.ui.coopGraphicsView.addItem(zero_line)

    def _update_trace_plots(self):
        for time_indicator_line in self._time_indicator_lines:
            time_indicator_line.setValue(self.sim_master.t / 1000.)

        for distance_indicator_line in self._distance_indicator_lines:
            distance_indicator_line.setValue(self.average_travelled_distance_trace[self.sim_master.time_index])

    def _update_live_plots(self, belief, belief_time_stamps, position_plan, action_plan, perceived_risk, risk_bounds, did_replanning,
                           belief_point_contributing_to_risk):
        self.ui.graphicsView.clear()

        plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']

        x = np.linspace(0., self.track.total_distance, 200)
        for belief_index, belief_point in enumerate(belief):
            mean, sd = belief_point
            if sd < 10e-6:
                sd = 10e-6
            color_index = belief_index
            while color_index >= len(plot_colors):
                color_index -= len(plot_colors)

            if belief_point_contributing_to_risk[belief_index]:
                color = plot_colors[color_index]
            else:
                color = 0.4

            self.ui.graphicsView.plot(x, stats.norm.pdf(x, mean, sd), pen=pyqtgraph.mkPen(color=color))

        belief_times_from_now = np.array(belief_time_stamps) - (self.sim_master.t / 1000.)
        position_indices = ((belief_times_from_now / (self.sim_master.dt / 1000.)) - 1).astype(int)

        for index, position in enumerate(position_plan[position_indices]):
            color_index = index
            while color_index >= len(plot_colors):
                color_index -= len(plot_colors)

            if belief_point_contributing_to_risk[index]:
                color = plot_colors[color_index]
            else:
                color = 0.4

            self.ui.graphicsView.plot([position], [0.0], pen=None, symbol='o', symbolBrush=pyqtgraph.mkBrush(color=color), symbolPen=None)

        self.ui.percievedRiskDoubleSpinBox.setValue(perceived_risk)
        self.ui.lowRiskBoundDoubleSpinBox.setValue(risk_bounds[0])
        self.ui.highRiskBoundDoubleSpinBox.setValue(risk_bounds[1])
        self.ui.replanningLineEdit.setText(str(did_replanning))
        self.ui.inputSlider.setValue(action_plan[0] * 100)

    def _enable_recording(self):
        if not self.is_recording:
            self.initialize_recording()
        else:
            self.stop_recording()

    def initialize_recording(self):
        file_name = datetime.datetime.now().strftime('-%Y%m%d-%Hh%Mm%Ss.avi')

        self.path_to_video_file = os.path.join('data', 'videos', file_name)
        fps = 1 / (self.sim_master.dt / 1000.)

        frame_size = self._get_image_of_current_gui().size()
        self.video_writer = cv2.VideoWriter(self.path_to_video_file, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), fps, (frame_size.width(), frame_size.height()))
        self.is_recording = True
        self.sim_master.enable_recording(True)

    def stop_recording(self):
        self.video_writer.release()
        QtWidgets.QMessageBox.information(self, 'Video Saved', 'A video capture of the visualisation was saved to ' + self.path_to_video_file)
        self.is_recording = False

    def record_frame(self):
        if self.is_recording:
            image = self._get_image_of_current_gui()
            frame_size = image.size()
            bits = image.bits()

            bits.setsize(frame_size.height() * frame_size.width() * 4)
            image_array = np.frombuffer(bits, np.uint8).reshape((frame_size.height(), frame_size.width(), 4))
            color_convert_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            self.video_writer.write(color_convert_image)

    def _save_screen_shot(self):
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        file_name = os.path.join('data', 'images', time_stamp + '.png')

        image = self._get_image_of_current_gui()
        image.save(file_name)

    def _get_image_of_current_gui(self):
        image = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
        region = QtGui.QRegion(self.rect())

        painter = QtGui.QPainter(image)
        self.render(painter, QtCore.QPoint(), region)
        painter.end()

        return image
