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
import pickle
import sys

from PyQt5 import QtWidgets

from agents import CEIAgent
from controllableobjects import PointMassObject
from gui import SimulationGui
from simulation.playback_master import PlaybackMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import SymmetricMergingTrack, StraightTrack
from trackobjects.trackside import TrackSide

simulation_constants: SimulationConstants

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    file_name = 'scenario_A.pkl'

    with open(os.path.join('data', file_name), 'rb') as f:
        playback_data = pickle.load(f)

    simulation_constants = playback_data['simulation_constants']

    dt = simulation_constants.dt  # ms
    vehicle_width = simulation_constants.vehicle_width
    vehicle_length = simulation_constants.vehicle_length

    if playback_data['agent_types'][TrackSide.LEFT] == CEIAgent:
        time_horizon = playback_data['belief_time_stamps'][TrackSide.LEFT][0][-2]
    else:
        time_horizon = 0

    try:
        track = playback_data['track']
    except KeyError:
        print('WARNING: the loaded simulation did not contain a track object (this is a deprecated format). A new track for playback will be created.')
        track = SymmetricMergingTrack(simulation_constants)

    try:
        surroundings = playback_data['surroundings']
    except KeyError:
        print('WARNING: the loaded simulation did not contain a surrounds object (this is a deprecated format). No surroundings will be displayed.')
        surroundings = None

    gui = SimulationGui(track, in_replay_mode=True, number_of_belief_points=int(time_horizon), surroundings=surroundings)
    sim_master = PlaybackMaster(gui, track, simulation_constants, playback_data)

    left_point_mass_object = PointMassObject(track, initial_position=track.get_start_position(TrackSide.LEFT), use_discrete_inputs=False)
    gui.add_controllable_car(left_point_mass_object, vehicle_length, vehicle_width, side_for_dial=TrackSide.LEFT, color='blue')
    sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object)

    if TrackSide.RIGHT in playback_data['agent_types'].keys():
        right_point_mass_object = PointMassObject(track, initial_position=track.get_start_position(TrackSide.RIGHT), use_discrete_inputs=False)
        gui.add_controllable_car(right_point_mass_object, vehicle_length, vehicle_width, side_for_dial=TrackSide.RIGHT, color='red')
        sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object)

    gui.register_sim_master(sim_master)
    sim_master.initialize_plots()
    sys.exit(app.exec_())
