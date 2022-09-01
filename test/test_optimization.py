from scipy import optimize

from agents import CEIAgent
from controllableobjects import PointMassObject
from simulation.simulationconstants import SimulationConstants
from test.fakesimmaster import FakeSimMaster
from trackobjects import SymmetricMergingTrack
from trackobjects.trackside import TrackSide

if __name__ == '__main__':
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50.,
                                               max_time=30e3)

    track = SymmetricMergingTrack(simulation_constants)

    lower_bound_threshold = track._lower_bound_threshold
    v0 = 10.
    x0 = lower_bound_threshold
    p0 = x0 - v0 * 3.0

    sim_master = FakeSimMaster(x0=x0 - v0 * 3.0, v0=v0)

    agent = CEIAgent(controllable_object=PointMassObject(track),
                     track_side=TrackSide.LEFT,
                     dt=simulation_constants.dt,
                     sim_master=sim_master,
                     track=track,
                     risk_bounds=(0.2, 0.5),
                     saturation_time=1.0,
                     vehicle_width=simulation_constants.vehicle_width,
                     vehicle_length=simulation_constants.vehicle_length,
                     preferred_velocity=10.,
                     time_horizon=4.,
                     belief_frequency=2,
                     theta=1.)

    a0 = agent.action_plan
    agent._initialize_belief()

    result = optimize.minimize(agent._cost_function,
                               a0,
                               args=(v0,
                                     0.0,
                                     0.0),
                               jac=agent.cost_jacobian,
                               bounds=agent.action_bounds,
                               options={'maxiter': 500},
                               constraints={'type': 'ineq',
                                            'fun': agent._plan_constraint,
                                            'args': (p0,
                                                     v0,
                                                     0.0,
                                                     0.0)})
    print(result)
