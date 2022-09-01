import random
import unittest

from scipy import optimize

from agents import CEIAgent
from controllableobjects import PointMassObject
from simulation.simulationconstants import SimulationConstants
from trackobjects import SymmetricMergingTrack
from trackobjects.trackside import TrackSide
from .fakesimmaster import FakeSimMaster


class test_CEI_jacobian(unittest.TestCase):
    def test_jacobian(self):
        section_length = random.uniform(10.0, 100.)
        start_point_distance = random.uniform(0.3 * section_length, 0.8 * section_length)

        vehicle_length = random.uniform(3., 8.)
        vehicle_width = random.uniform(vehicle_length / 2., vehicle_length)

        simulation_constants = SimulationConstants(dt=50,
                                                   vehicle_width=vehicle_width,
                                                   vehicle_length=vehicle_length,
                                                   track_start_point_distance=start_point_distance,
                                                   track_section_length=section_length,
                                                   max_time=30e3)

        track = SymmetricMergingTrack(simulation_constants)
        sim_master = FakeSimMaster()

        controllable_object = PointMassObject(track, use_discrete_inputs=False)

        agent = CEIAgent(controllable_object, TrackSide.LEFT, simulation_constants.dt, sim_master, track, risk_bounds=(0.15, 0.3), saturation_time=1.,
                         time_horizon=4.,
                         preferred_velocity=10.,
                         vehicle_width=vehicle_width, vehicle_length=vehicle_length,
                         theta=1., belief_frequency=4)

        for t in range(200):
            controllable_object.set_continuous_acceleration(agent.compute_continuous_input(simulation_constants.dt / 1000.0))
            controllable_object.update_model(simulation_constants.dt / 1000.0)
            sim_master.update(simulation_constants.dt)

            grad_check_result = optimize.check_grad(agent._cost_function, agent.cost_jacobian,
                                                    agent.action_plan,
                                                    controllable_object.velocity,
                                                    controllable_object.resistance_coefficient,
                                                    controllable_object.constant_resistance)
            self.assertTrue(abs(grad_check_result) < 10e-05, 'difference between the Jacobian and the estimated gradient should be smaller then 10e-5, it is '
                                                             'currently %f' % abs(grad_check_result))
