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
import tqdm

from agents import CEIAgent
from simulation.abstractsimmaster import AbstractSimMaster
from trackobjects.trackside import TrackSide


class OfflineSimMaster(AbstractSimMaster):
    def __init__(self, track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=True):
        super().__init__(track, simulation_constants, file_name, save_to_mat_and_csv=save_to_mat_and_csv)
        self.verbose = verbose

        if verbose:
            self._progress_bar = tqdm.tqdm()
        else:
            self._progress_bar = None

        self._stop = False

    def add_vehicle(self, side: TrackSide, controllable_object, agent):
        self._vehicles[side] = controllable_object
        self._agents[side] = agent
        self.agent_types[side] = type(agent)

        if type(agent) == CEIAgent:
            self.risk_bounds[side] = agent.risk_bounds
        else:
            self.risk_bounds[side] = None

    def start(self):
        self._store_current_status()

        while self.t <= self.max_time and not self._stop:
            self.do_time_step()
            self._t += self.dt
            self.time_index += 1
            if self.verbose:
                self._progress_bar.update()

        if not self._stop:
            self.end_state = "Time ran out"

        self._save_to_file()

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
                self.end_state = "Beyond track bounds"
                self._stop = True
            elif self._track.is_beyond_finish(controllable_object.position):
                self.end_state = "Finished"
                self._stop = True

        lb, ub = self._track.get_collision_bounds(self._vehicles[TrackSide.LEFT].traveled_distance, self.vehicle_width, self.vehicle_length, )
        if lb is not None and ub is not None:
            try:
                if lb <= self._vehicles[TrackSide.RIGHT].traveled_distance <= ub:
                    self.end_state = "Collided"
                    self._stop = True
            except KeyError:
                # no right side vehicle exists
                pass

        self._store_current_status()
