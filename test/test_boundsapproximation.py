import random
import unittest

import numpy as np
import tqdm

from simulation.simulationconstants import SimulationConstants
from trackobjects import SymmetricMergingTrack


class TestCollisionBoundaries(unittest.TestCase):
    def test_bounds_approximation(self):
        section_length = random.uniform(10.0, 100.)
        start_point_distance = random.uniform(0.3 * section_length, 0.8 * section_length)

        vehicle_length = random.uniform(3., 8.)
        vehicle_width = random.uniform(vehicle_length / 2., vehicle_length)

        # vehicle_width = 1.8
        # vehicle_length = 4.5
        # start_point_distance = 25.
        # section_length = 50.

        print('section length = %.2f' % section_length)
        print('start point distance = %.2f' % start_point_distance)
        print('vehicle length = %.2f' % vehicle_length)
        print('vehicle width = %.2f' % vehicle_width)

        simulation_constants = SimulationConstants(dt=50,
                                                   vehicle_width=vehicle_width,
                                                   vehicle_length=vehicle_length,
                                                   track_start_point_distance=start_point_distance,
                                                   track_section_length=section_length,
                                                   max_time=30e3)

        track = SymmetricMergingTrack(simulation_constants)

        track._initialize_linear_bound_approximation(simulation_constants.vehicle_width, simulation_constants.vehicle_length)

        # cm resolution lookup
        entries = [i for i in range(int(2 * simulation_constants.track_section_length * 100))]

        look_up_table = np.zeros((len(entries), 2))
        approximation_table = np.zeros((len(entries), 2))

        for entry in tqdm.tqdm(entries):
            travelled_distance = entry / 100.
            look_up_table[entry, :] = track.get_collision_bounds(travelled_distance, simulation_constants.vehicle_width, simulation_constants.vehicle_length)
            approximation_table[entry, :] = track.get_collision_bounds_approximation(travelled_distance)

        print('table constructed')

        errors = abs(look_up_table - approximation_table)

        print('max lower bound error = %.3f' % np.nanmax(errors[:, 0]))
        print('max upper bound error = %.3f' % np.nanmax(errors[:, 1]))

        self.assertTrue(np.nanmax(errors[:, 0]) <= 0.50, 'maximum error on the lower collision bound should be smaller than 50 cm')
        self.assertTrue(np.nanmax(errors[:, 1]) <= 0.50, 'maximum error on the upper collision bound should be smaller than 50 cm')
