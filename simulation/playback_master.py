import numpy as np

from agents import CEIAgent
from controllableobjects.controlableobject import ControllableObject
from simulation.simmaster import SimMaster
from trackobjects.trackside import TrackSide


class PlaybackMaster(SimMaster):
    def __init__(self, gui, track, simulation_constants, playback_data):
        super().__init__(gui, track, simulation_constants, file_name=None)

        self.playback_data = playback_data
        self.maxtime_index = len([p for p in playback_data['positions'][TrackSide.LEFT] if p is not None]) - 1

    def add_vehicle(self, side: TrackSide, controllable_object: ControllableObject, agent=None):
        super(PlaybackMaster, self).add_vehicle(side, controllable_object, agent=None)

        self._vehicles[side].position = self.playback_data['positions'][side][0]
        self._vehicles[side].velocity = self.playback_data['velocities'][side][0]

        self.gui.update_all_graphics(left_velocity=self.playback_data['velocities'][TrackSide.LEFT][0],
                                     right_velocity=self.playback_data['velocities'][TrackSide.RIGHT][0])

    def set_time(self, time_promille):
        new_index = int((time_promille / 1000.) * self.maxtime_index)
        self.time_index = new_index - 1
        self._t = self.time_index * self.dt
        self.do_time_step()

    def initialize_plots(self):
        time = [(self.playback_data['dt'] / 1000.) * index for index in range(len(self.playback_data['travelled_distance'][TrackSide.LEFT]))]
        plot_live_plot = self.playback_data['agent_types'][TrackSide.LEFT] == CEIAgent
        self.gui.initialize_plots(self.playback_data, time, 'b', 'r', plot_live_plot=plot_live_plot)

    def do_time_step(self, reverse=False):
        if reverse and self.time_index > 0:
            self._t -= self.dt
            self.time_index -= 1
            self.gui.show_overlay()
        elif not reverse and self.time_index < self.maxtime_index:
            self._t += self.dt
            self.time_index += 1
            self.gui.show_overlay()
        elif not reverse:
            if self.main_timer.isActive():
                self.gui.toggle_play()
            self.gui.show_overlay(self.playback_data['end_state'])
            if self._is_recording:
                self.gui.record_frame()
                self.gui.stop_recording()
            return

        self.gui.update_time_label(self.t / 1000.0)

        for side in self._vehicles.keys():
            self._vehicles[side].position = self.playback_data['positions'][side][self.time_index]
            self._vehicles[side].velocity = self.playback_data['velocities'][side][self.time_index]

        self.gui.update_all_graphics(left_velocity=self.playback_data['velocities'][TrackSide.LEFT][self.time_index],
                                     right_velocity=self.playback_data['velocities'][TrackSide.RIGHT][self.time_index])

        if self.playback_data['agent_types'][TrackSide.LEFT] == CEIAgent:
            self.gui.update_plots(self.playback_data['beliefs'][TrackSide.LEFT][self.time_index][0:-1],
                                  self.playback_data['belief_time_stamps'][TrackSide.LEFT][self.time_index][0:-1],
                                  self.playback_data['position_plans'][TrackSide.LEFT][self.time_index],
                                  self.playback_data['action_plans'][TrackSide.LEFT][self.time_index],
                                  self.playback_data['perceived_risks'][TrackSide.LEFT][self.time_index],
                                  self.playback_data['risk_bounds'][TrackSide.LEFT],
                                  self.playback_data['is_replanning'][TrackSide.LEFT][self.time_index],
                                  self.playback_data['belief_point_contributing_to_risk'][TrackSide.LEFT][self.time_index])

        if self._is_recording:
            self.gui.record_frame()
