import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from agents.ceiagent import CEIAgent

if __name__ == '__main__':
    time_horizon = 4.
    belief_frequency = 4.
    belief_dt = 1/belief_frequency
    update_dt = 0.05
    belief_length = int(time_horizon * belief_frequency)
    total_track_distance = 40

    #  Initial belief
    belief = []
    for n in range(belief_length):
        belief.append([0.0, 0.0])
    belief_time_stamps = [belief_dt * (n + 1) for n in range(belief_length)]
    max_acceleration = 2.5
    max_comfortable_acceleration = 1.5

    initial_other_position = 0.
    other_constant_velocity = 10.
    other_initial_acceleration = 0.1

    upper_velocity_bound = lower_velocity_bound = other_constant_velocity
    upper_position_bound = lower_position_bound = initial_other_position

    for belief_index in range(belief_length):
        upper_position_bound += upper_velocity_bound * (1 / belief_frequency) + (max_acceleration / 2.) * (
                1 / belief_frequency) ** 2
        upper_velocity_bound += max_acceleration * (1 / belief_frequency)

        new_lower_position_bound = lower_position_bound + lower_velocity_bound * (1 / belief_frequency) + (
                -max_acceleration / 2.) * (
                                           1 / belief_frequency) ** 2
        if new_lower_position_bound >= lower_position_bound:
            lower_position_bound = new_lower_position_bound

        lower_velocity_bound -= max_acceleration * (1 / belief_frequency)

        if lower_velocity_bound < 0.:
            lower_velocity_bound = 0.

        mean = ((upper_position_bound - lower_position_bound) / 2.) + lower_position_bound
        sd = (upper_position_bound - mean) / 3

        belief[belief_index][0] = mean
        belief[belief_index][1] = sd
        belief_time_stamps.append((1 / belief_frequency) * (belief_index + 1))

    #  updated belief
    other_position = initial_other_position + other_constant_velocity * update_dt

    new_belief = []
    samples = np.array(other_constant_velocity)

    for belief_point_index in range(belief_length):
        prior_mu, prior_sigma = belief[belief_point_index]
        prior_mu -= other_position

        time = belief_time_stamps[belief_point_index] - update_dt
        likelihood_sigma = (max_comfortable_acceleration * time) / 6

        posterior_mu, posterior_sigma = CEIAgent._calculate_posterior(prior_mu, prior_sigma, likelihood_sigma, samples, time)
        posterior_mu += other_position

        new_belief += [[posterior_mu, posterior_sigma]]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 5))

    plan_points = [8., 10.5, 13., 15.5]
    vehicle_length = 4.5
    lower_collision_bounds = np.array(plan_points) - vehicle_length
    upper_collision_bounds = np.array(plan_points) + vehicle_length

    x = np.linspace(0., total_track_distance, 2000)
    colors = ['tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for index, (belief_point, updated_belief_point) in enumerate(zip(belief[-12:-8], new_belief[-12:-8])):
        mean, sd = belief_point

        distribution = stats.norm.pdf(x, mean, sd)
        belief_handle, = axes[0].plot(x[distribution > 0.001], distribution[distribution > 0.001], c=colors[index], label='Initial believe')

        mean, sd = updated_belief_point
        distribution = stats.norm.pdf(x, mean, sd)
        axes[1].plot(x[distribution > 0.001], distribution[distribution > 0.001], c=colors[index], label='Updated believe')

        plan_handle = axes[0].scatter(plan_points[index], [0.], c=colors[index], label='Planned own vehicle center position', zorder=10)
        axes[1].scatter(plan_points[index], [0.], c=colors[index], label='plan', zorder=10)

    mean, sd = new_belief[-10]
    distribution = stats.norm.pdf(x, mean, sd)
    axes[2].plot(x[distribution > 0.001], distribution[distribution > 0.001], c=colors[2], label='Updated believe')
    axes[2].vlines(lower_collision_bounds[-2], 0., 4., colors='grey', linestyles='dashed')
    bounds_handle = axes[2].vlines(upper_collision_bounds[-2], 0., 4., colors='grey', linestyles='dashed')
    axes[2].scatter(plan_points[-2], [0.], c=colors[2], label='plan', zorder=10)

    x = np.linspace(lower_collision_bounds[-2], upper_collision_bounds[-2], 100)
    distribution = stats.norm.pdf(x, mean, sd)
    axes[2].fill_between(x[distribution > 0.001], distribution[distribution > 0.001], color=colors[2])
    axes[2].annotate(r'$p_c = 0.5$', [17., 0.5], bbox=dict(boxstyle="round,pad=0.3", fc="#EEEEEE", ec='grey', lw=2), arrowprops={'width': 1, 'headwidth': 5, 'facecolor':'black'},
                     xytext=(14., 1.))
    axes[2].arrow(13.7, .5, -1., 0., width=.1, color='k', alpha=0.7)

    axes[0].set_xlim((6., 25.))
    for ax in axes:
        ax.set_ylim((-.5, 2.))

    axes[0].legend([belief_handle, plan_handle], ['Believed other vehicle center position', 'Planned own vehicle center position'])
    axes[2].legend([bounds_handle], ['Bounds of Collision'])
    axes[2].set_xlabel('Lateral position along the track [m]')

    titles = ['a)', 'b)', 'c)']
    for index, ax in enumerate(axes):
        ax.hlines(0., 0., 30., colors='lightgrey', linewidths=1., zorder=-1)
        ax.set_ylabel('Probability \n density')
        ax.set_title(titles[index], x=-0.12, y=0.45)

    plt.show()
