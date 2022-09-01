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
import glob
import multiprocessing as mp
import os

from agents import CEIAgent
from controllableobjects import PointMassObject
from simulation.offlinesimmaster import OfflineSimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import StraightTrack
from trackobjects.trackside import TrackSide


def simulate(follower_velocity, track, simulation_constants):
    file_name = 'velocity_%.1f' % follower_velocity
    sim_master = OfflineSimMaster(track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=False)

    v_left = follower_velocity
    v_right = follower_velocity * 0.9

    left_point_mass_object = PointMassObject(track,
                                             initial_position=track.traveled_distance_to_coordinates(0.0),
                                             initial_velocity=v_left,
                                             cruise_control_velocity=v_left,
                                             use_discrete_inputs=False,
                                             resistance_coefficient=0.0005, constant_resistance=0.1)

    right_point_mass_object = PointMassObject(track,
                                              initial_position=track.traveled_distance_to_coordinates(follower_velocity +
                                                                                                      simulation_constants.vehicle_length),
                                              initial_velocity=v_right,
                                              cruise_control_velocity=v_right,
                                              use_discrete_inputs=False,
                                              resistance_coefficient=0.0005, constant_resistance=0.1)

    cei_agent_1 = CEIAgent(left_point_mass_object, TrackSide.LEFT, simulation_constants.dt, sim_master, track,
                           risk_bounds=(.2, .5),
                           saturation_time=2.,
                           time_horizon=4.,
                           preferred_velocity=v_left,
                           vehicle_width=simulation_constants.vehicle_width,
                           belief_frequency=4,
                           vehicle_length=simulation_constants.vehicle_length,
                           theta=1.)

    cei_agent_2 = CEIAgent(right_point_mass_object, TrackSide.RIGHT, simulation_constants.dt, sim_master, track,
                           risk_bounds=(.2, .5),
                           saturation_time=2.,
                           time_horizon=4.,
                           preferred_velocity=v_right,
                           vehicle_width=simulation_constants.vehicle_width,
                           belief_frequency=4,
                           vehicle_length=simulation_constants.vehicle_length,
                           theta=1.)

    sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object, cei_agent_1)
    sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object, cei_agent_2)

    sim_master.start()

    print('')
    print('simulation ended with exit status: ' + sim_master.end_state)


if __name__ == '__main__':
    os.chdir(os.getcwd() + '\\..')

    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=10.,
                                               track_section_length=200.,
                                               max_time=40e3)

    track = StraightTrack(simulation_constants)
    follower_velocities = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    all_files = [os.path.basename(f) for f in glob.glob(os.path.join('data', 'velocity_*.pkl'))]

    with mp.Pool(2) as p:
        length = len(follower_velocities)
        p.starmap(simulate, zip(follower_velocities, [track] * length, [simulation_constants] * length))
