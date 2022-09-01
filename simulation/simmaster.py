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
from PyQt5 import QtCore

from agents import CEIAgent
from agents.agent import Agent
from controllableobjects.controlableobject import ControllableObject
from simulation.abstractsimmaster import AbstractSimMaster
from trackobjects.trackside import TrackSide


class SimMaster(AbstractSimMaster):
    def __init__(self, gui, track, simulation_constants, *, file_name=None, sub_folder=None, save_to_mat_and_csv=True):
        super().__init__(track, simulation_constants, file_name, sub_folder=sub_folder, save_to_mat_and_csv=save_to_mat_and_csv)

        self.main_timer = QtCore.QTimer()
        self.main_timer.setInterval(self.dt)
        self.main_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.main_timer.setSingleShot(False)
        self.main_timer.timeout.connect(self.do_time_step)

        self.count_down_timer = QtCore.QTimer()
        self.count_down_timer.setInterval(1000)
        self.count_down_timer.timeout.connect(self.count_down)

        self.count_down_clock = 3  # counts down from 3
        self.history_length = 5

        self.gui = gui

        self.velocity_history = {TrackSide.LEFT: [],
                                 TrackSide.RIGHT: []}

    def start(self):
        self._store_current_status()
        self.count_down()
        self.count_down_timer.start()

    def pause(self):
        self.main_timer.stop()

    def count_down(self):
        if self.count_down_clock == 0:
            self.main_timer.start()
            self.count_down_timer.stop()
            self.gui.show_overlay()
        else:
            self.gui.show_overlay(str(self.count_down_clock))
            self.count_down_clock -= 1

    def add_vehicle(self, side: TrackSide, controllable_object: ControllableObject, agent: Agent):
        self._vehicles[side] = controllable_object
        self._agents[side] = agent
        self.agent_types[side] = type(agent)

        self.velocity_history[side] = [controllable_object.velocity] * self.history_length

    def get_velocity_history(self, side: TrackSide):
        return self.velocity_history[side]

    def _update_history(self):
        for side in TrackSide:
            try:
                self.velocity_history[side] = [self._vehicles[side].velocity] + self.velocity_history[side][:-1]
            except KeyError:
                # no vehicle exists on that side
                pass

    def _end_simulation(self):
        self.gui.show_overlay(self.end_state)
        self._save_to_file()

        if self._is_recording:
            self.gui.record_frame()
            self.gui.stop_recording()

    def do_time_step(self, reverse=False):

        for controllable_object, agent in zip(self._vehicles.values(), self._agents.values()):
            if controllable_object.use_discrete_inputs:
                controllable_object.set_discrete_acceleration(agent.compute_discrete_input(self.dt / 1000.0))
            else:
                controllable_object.set_continuous_acceleration(agent.compute_continuous_input(self.dt / 1000.0))

        # This for loop over agents is done twice because the models that compute the new input need the current state of other vehicles.
        # So plan first for all vehicles before applying the accelerations and calculating the new state
        for controllable_object, agent in zip(self._vehicles.values(), self._agents.values()):
            controllable_object.update_model(self.dt / 1000.0)

            if self._track.is_beyond_track_bounds(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Beyond track bounds"
            elif self._track.is_beyond_finish(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Finished"

        lb, ub = self._track.get_collision_bounds(self._vehicles[TrackSide.LEFT].traveled_distance, self.vehicle_width, self.vehicle_length, )
        if lb and ub:
            try:
                if lb <= self._vehicles[TrackSide.RIGHT].traveled_distance <= ub:
                    self.main_timer.stop()
                    self.end_state = "Collided"
            except KeyError:
                # no right side vehicle exists
                pass

        self._update_history()

        try:
            self.gui.update_all_graphics(left_velocity=self._vehicles[TrackSide.LEFT].velocity,
                                         right_velocity=self._vehicles[TrackSide.RIGHT].velocity)
        except KeyError:
            # no right side vehicle
            self.gui.update_all_graphics(left_velocity=self._vehicles[TrackSide.LEFT].velocity,
                                         right_velocity=0.0)

        cei_agent = self._agents[TrackSide.LEFT]
        if isinstance(cei_agent, CEIAgent):
            self.gui.update_plots(cei_agent.belief[:-1], cei_agent.belief_time_stamps[:-1], cei_agent.position_plan, cei_agent.action_plan,
                                  cei_agent.perceived_risk, cei_agent.risk_bounds, cei_agent.did_plan_update_on_last_tick,
                                  cei_agent.belief_point_contributing_to_risk)
        self.gui.update_time_label(self.t / 1000.0)
        self._t += self.dt
        self.time_index += 1
        self._store_current_status()

        if self._t >= self.max_time:
            self.end_state = self.end_state = "Time ran out"

        if self.end_state != 'Not finished':
            self._end_simulation()

        if self._is_recording:
            self.gui.record_frame()
