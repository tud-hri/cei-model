import copy
import warnings

import autograd
import autograd.numpy as np
from autograd.scipy import stats
from scipy import optimize

from controllableobjects import ControllableObject
from trackobjects.trackside import TrackSide
from .agent import Agent


class CEIAgent(Agent):
    """
    An agent used in a Communication-Enabled Interaction model
    """

    def __init__(self, controllable_object: ControllableObject, track_side: TrackSide, dt, sim_master, track, risk_bounds, saturation_time, vehicle_width,
                 vehicle_length, preferred_velocity, time_horizon, belief_frequency, theta):
        self.controllable_object = controllable_object
        self.track_side = track_side
        self.dt = dt
        self.sim_master = sim_master
        self.track = track
        self.risk_bounds = risk_bounds
        self.theta = theta
        self.saturation_time = saturation_time
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.preferred_velocity = preferred_velocity
        self.time_horizon = time_horizon
        self.belief_frequency = belief_frequency

        # the action plan consists of the action (acceleration) to take at the coming time steps. The position plan is the set of positions along the track
        # where the ego vehicle will end up when taking these actions.
        self.action_plan = np.array([0.] * int((1000 / dt) * time_horizon))
        self.velocity_plan = np.array([0.0] * int((1000 / dt) * time_horizon))
        self.position_plan = np.array([0.0] * int((1000 / dt) * time_horizon))
        self.action_bounds = optimize.Bounds([-1.] * len(self.action_plan), [1.] * len(self.action_plan))

        # the belief consists of sets of a mean and standard deviation for a distribution over positions at every time step.
        self.belief = []
        self.belief_time_stamps = []
        self.belief_point_contributing_to_risk = []
        for belief_index in range(int(belief_frequency * time_horizon) + 1):
            self.belief.append([0., 0.])

        self._time_of_last_update = 0.0
        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.

        # the sigma of the likelihood function is fixed and assumed based on the bound of comfortable acceleration (Hoberock 1977)
        self.max_comfortable_acceleration = 1.

        # The observed communication is the current velocity of the other vehicle
        self.observed_communication = 0.0

        self.cost_jacobian = autograd.jacobian(self._cost_function)
        self._is_initialized = False

    def reset(self):
        self.action_plan = np.array([0.] * int((1000 / self.dt) * self.time_horizon))
        self.velocity_plan = np.array([0.0] * int((1000 / self.dt) * self.time_horizon))
        self.position_plan = np.array([0.0] * int((1000 / self.dt) * self.time_horizon))
        self.action_bounds = optimize.Bounds([-1.] * len(self.action_plan), [1.] * len(self.action_plan))

        # the belief consists of sets of a mean and standard deviation for a distribution over positions at every time step.
        self.belief = []
        self.belief_time_stamps = []
        self.belief_point_contributing_to_risk = []
        for belief_index in range(int(self.belief_frequency * self.time_horizon) + 1):
            self.belief.append([0., 0.])

        self._time_of_last_update = 0.0
        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.

        # The observed communication is the current velocity of the other vehicle
        self.observed_communication = 0.0
        self._is_initialized = False

    def _observe_communication(self):
        _, other_velocity = self.sim_master.get_current_state(self.track_side.other)

        self.observed_communication = other_velocity

    def _initialize_belief(self):
        other_position, other_velocity = self.sim_master.get_current_state(self.track_side.other)

        if other_position is None or other_velocity is None:
            # no other vehicle exists, this can be approximated by assuming the other vehicle is stationary at 0.0
            other_position = 0.0
            other_velocity = 0.0

        upper_velocity_bound = lower_velocity_bound = other_velocity
        upper_position_bound = lower_position_bound = other_position

        for belief_index in range(len(self.belief)):
            upper_position_bound += upper_velocity_bound * (1 / self.belief_frequency) + (self.controllable_object.max_acceleration / 2.) * (
                    1 / self.belief_frequency) ** 2
            upper_velocity_bound += self.controllable_object.max_acceleration * (1 / self.belief_frequency)

            new_lower_position_bound = lower_position_bound + lower_velocity_bound * (1 / self.belief_frequency) + (
                    -self.controllable_object.max_acceleration / 2.) * (
                                               1 / self.belief_frequency) ** 2
            if new_lower_position_bound >= lower_position_bound:
                lower_position_bound = new_lower_position_bound

            lower_velocity_bound -= self.controllable_object.max_acceleration * (1 / self.belief_frequency)

            if lower_velocity_bound < 0.:
                lower_velocity_bound = 0.

            mean = ((upper_position_bound - lower_position_bound) / 2.) + lower_position_bound
            sd = (upper_position_bound - mean) / 3

            self.belief[belief_index][0] = mean
            self.belief[belief_index][1] = sd
            self.belief_time_stamps.append((1 / self.belief_frequency) * (belief_index + 1))

    def _update_belief(self, generate_new_point):
        other_position, other_velocity = self.sim_master.get_current_state(self.track_side.other)
        time_step = 1 / self.belief_frequency

        if other_position is None or other_velocity is None:
            # no other vehicle exists, no only update the time stamps if needed
            if generate_new_point:
                self.belief_time_stamps = self.belief_time_stamps[1:] + [self.belief_time_stamps[-1] + time_step]
            return

        new_belief = []
        samples = np.array(self.observed_communication)

        first_index_to_consider = 1 if generate_new_point else 0

        for belief_point_index in range(first_index_to_consider, len(self.belief)):
            prior_mu, prior_sigma = self.belief[belief_point_index]
            prior_mu -= other_position

            time = self.belief_time_stamps[belief_point_index] - (self.sim_master.t / 1000.)
            likelihood_sigma = (self.max_comfortable_acceleration * time) / 6

            posterior_mu, posterior_sigma = self._calculate_posterior(prior_mu, prior_sigma, likelihood_sigma, samples, time)
            posterior_mu += other_position

            new_belief += [[posterior_mu, posterior_sigma]]

        if generate_new_point:
            # calculate bounds on end point
            time_until_last_point = time_step * len(self.belief)
            max_acceleration = self.controllable_object.max_acceleration
            min_velocity = other_velocity - (max_acceleration * time_until_last_point) / 2
            max_velocity = other_velocity + (max_acceleration * time_until_last_point) / 2

            if min_velocity < 0.:
                min_velocity = 0.

            lower_position_bound = other_position + min_velocity * time_until_last_point
            upper_position_bound = other_position + max_velocity * time_until_last_point

            last_mu = lower_position_bound + (upper_position_bound - lower_position_bound) / 2
            last_sigma = (upper_position_bound - last_mu) / 3

            new_belief += [[last_mu, last_sigma]]

            self.belief = new_belief
            self.belief_time_stamps = self.belief_time_stamps[1:] + [self.belief_time_stamps[-1] + time_step]
        else:
            self.belief = new_belief

    @staticmethod
    def _calculate_posterior(prior_mu, prior_sigma, likelihood_sigma, samples: np.ndarray, time_step):
        samples = np.array([samples])

        n = len(samples)

        posterior_sigma = (likelihood_sigma ** 2 * prior_sigma ** 2) / (likelihood_sigma ** 2 + prior_sigma ** 2 * (n / (time_step ** 2)))
        posterior_mu = (prior_mu * likelihood_sigma ** 2 + sum(samples * prior_sigma ** 2 / time_step)) / (
                likelihood_sigma ** 2 + prior_sigma ** 2 * (n / (time_step ** 2)))

        posterior_sigma = max(posterior_sigma, 1e-3)
        return posterior_mu, posterior_sigma

    def _evaluate_risk(self):
        max_risk, risk_per_point = self._get_collision_probability(self.belief, self.position_plan)
        self.belief_point_contributing_to_risk = [bool(p) for p in risk_per_point]
        return max_risk

    def _get_collision_probability(self, belief, position_plan):
        probabilities_over_plan = []
        current_time = self.sim_master.t / 1000.

        for belief_index, belief_point in enumerate(belief[0:-1]):
            time_from_now = self.belief_time_stamps[belief_index] - current_time

            assert abs(round(time_from_now / (self.dt / 1000)) - time_from_now / (self.dt / 1000)) < 10e-10

            plan_index = int(time_from_now / (self.dt / 1000)) - 1

            position_plan_point = position_plan[plan_index]
            lower_bound, upper_bound = self.track.get_collision_bounds_approximation(position_plan_point)

            if lower_bound and upper_bound:
                collision_probability = self._get_normal_probability(belief_point[0], belief_point[1], lower_bound, upper_bound)
                probabilities_over_plan += [collision_probability]
            else:
                probabilities_over_plan += [0.]

        return np.amax(probabilities_over_plan), probabilities_over_plan

    def _plan_constraint(self, plan, initial_position, initial_velocity, resistance_coefficient, constant_resistance):
        position_plan = []

        position = initial_position
        velocity = initial_velocity

        for index, acceleration_command in enumerate(plan):
            acceleration = acceleration_command * self.controllable_object.max_acceleration
            position, velocity = self.controllable_object.calculate_time_step_1d(self.dt / 1000., position, velocity, acceleration, resistance_coefficient,
                                                                                 constant_resistance)
            position_plan += [position]

        collision_probability, _ = self._get_collision_probability(self.belief, position_plan)

        return ((self.risk_bounds[0] + self.risk_bounds[1]) / 2) - collision_probability

    @staticmethod
    def _get_normal_probability(mu, sigma, lower_bound, upper_bound):
        if lower_bound is None:
            return stats.norm.cdf(upper_bound, mu, sigma)
        elif upper_bound is None:
            return 1 - stats.norm.cdf(lower_bound, mu, sigma)
        else:
            return stats.norm.cdf(upper_bound, mu, sigma) - stats.norm.cdf(lower_bound, mu, sigma)

    def _do_rough_grid_search_for_initial_condition(self):
        initial_conditions = [np.array([-1.] * len(self.action_plan)),
                              np.array([0.] * len(self.action_plan)),
                              np.array([1.] * len(self.action_plan)),
                              self.action_plan]

        last_cost = np.inf
        last_constraint = -np.inf
        initial_condition_to_use = None
        initial_condition_with_highest_constraint = None

        for initial_condition in initial_conditions:
            constraint = self._plan_constraint(initial_condition, self.controllable_object.traveled_distance,
                                               self.controllable_object.velocity,
                                               self.controllable_object.resistance_coefficient,
                                               self.controllable_object.constant_resistance)

            if constraint > last_constraint:
                initial_condition_with_highest_constraint = initial_condition
                last_constraint = constraint

            if constraint >= 0.:
                cost = self._cost_function(initial_condition,
                                           self.controllable_object.velocity,
                                           self.controllable_object.resistance_coefficient,
                                           self.controllable_object.constant_resistance)

                if cost < last_cost:
                    initial_condition_to_use = initial_condition
                    last_cost = cost

        if initial_condition_to_use is None:
            return initial_condition_with_highest_constraint

        return initial_condition_to_use

    def _update_plan(self):

        result = optimize.minimize(self._cost_function,
                                   self.action_plan,
                                   args=(self.controllable_object.velocity,
                                         self.controllable_object.resistance_coefficient,
                                         self.controllable_object.constant_resistance),
                                   jac=self.cost_jacobian,
                                   bounds=self.action_bounds,
                                   constraints={'type': 'ineq',
                                                'fun': self._plan_constraint,
                                                'args': (self.controllable_object.traveled_distance,
                                                         self.controllable_object.velocity,
                                                         self.controllable_object.resistance_coefficient,
                                                         self.controllable_object.constant_resistance)})

        if not result.success:
            initial_condition = self._do_rough_grid_search_for_initial_condition()
            result = optimize.minimize(self._cost_function,
                                       initial_condition,
                                       args=(self.controllable_object.velocity,
                                             self.controllable_object.resistance_coefficient,
                                             self.controllable_object.constant_resistance),
                                       jac=self.cost_jacobian,
                                       bounds=self.action_bounds,
                                       constraints={'type': 'ineq',
                                                    'fun': self._plan_constraint,
                                                    'args': (self.controllable_object.traveled_distance,
                                                             self.controllable_object.velocity,
                                                             self.controllable_object.resistance_coefficient,
                                                             self.controllable_object.constant_resistance)})
            if not result.success:
                warnings.warn('planning failed')

        self.action_plan = result.x
        self._calculate_position_plan()

    def _cost_function(self, plan, initial_velocity, resistance_coefficient, constant_resistance):
        velocities = [0.] * len(plan)
        previous_position = 0.
        previous_velocity = initial_velocity

        for index, acceleration_command in enumerate(plan):
            acceleration = acceleration_command * self.controllable_object.max_acceleration
            previous_position, previous_velocity = self.controllable_object.calculate_time_step_1d(self.dt / 1000., previous_position, previous_velocity,
                                                                                                   acceleration,
                                                                                                   resistance_coefficient, constant_resistance)
            velocities[index] = previous_velocity

        velocities = np.array(velocities)
        cost = sum((velocities - self.preferred_velocity) ** 2 + self.theta * plan ** 2)
        return cost

    def _calculate_position_plan(self):
        previous_position = self.controllable_object.traveled_distance
        previous_velocity = copy.copy(self.controllable_object.velocity)

        for index, acceleration_command in enumerate(self.action_plan):
            acceleration = acceleration_command * self.controllable_object.max_acceleration
            previous_position, previous_velocity = self.controllable_object.calculate_time_step_1d(self.dt / 1000., previous_position, previous_velocity,
                                                                                                   acceleration,
                                                                                                   self.controllable_object.resistance_coefficient,
                                                                                                   self.controllable_object.constant_resistance)
            self.velocity_plan[index] = previous_velocity
            self.position_plan[index] = previous_position

    def _continue_current_plan(self):
        self.action_plan = np.roll(self.action_plan, -1)

        target_velocity = self.velocity_plan[-1]
        required_acceleration = self.controllable_object.resistance_coefficient * target_velocity ** 2 + self.controllable_object.constant_resistance

        self.action_plan[-1] = required_acceleration / self.controllable_object.max_acceleration

        self._calculate_position_plan()

    def _convert_plan_to_communicative_action(self):
        pass

    def compute_discrete_input(self, dt):
        pass

    def compute_continuous_input(self, dt):
        if not self._is_initialized:
            self._initialize_belief()
            self._update_plan()
            self._evaluate_risk()
            self._is_initialized = True
        else:
            self._observe_communication()
            self._update_belief(generate_new_point=self.sim_master.t % (1000 / self.belief_frequency) == 0.)

            self._continue_current_plan()
            self.perceived_risk = self._evaluate_risk()

            if not self.controllable_object.cruise_control_active:
                if self.perceived_risk < self.risk_bounds[0] and (self.sim_master.t / 1000) - self._time_of_last_update > self.saturation_time:
                    self._time_of_last_update = self.sim_master.t / 1000
                    self.did_plan_update_on_last_tick = -1
                    self._update_plan()
                    self.perceived_risk = self._evaluate_risk()
                elif self.perceived_risk > self.risk_bounds[1]:
                    self._time_of_last_update = self.sim_master.t / 1000
                    self.did_plan_update_on_last_tick = 1
                    self._update_plan()
                    self.perceived_risk = self._evaluate_risk()
                else:
                    self.did_plan_update_on_last_tick = 0

        return self.action_plan[0]

    @property
    def name(self):
        pass
