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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from trackobjects.trackside import TrackSide


def plot_trial(data, title='', plot_gap=True, mark_replans=False):
    freq = int(1000 / data['dt'])
    plot_risks = data['perceived_risks'][TrackSide.LEFT] and data['perceived_risks'][TrackSide.RIGHT]

    figure = plt.figure(figsize=(8, 9))
    figure.suptitle(title)

    time = [t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))]
    titles = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    title_iterator = titles.__iter__()

    if plot_risks and plot_gap:
        grid = plt.GridSpec(6, 1, wspace=0.1, hspace=0.7)
    elif plot_risks or plot_gap:
        grid = plt.GridSpec(5, 1, wspace=0.1, hspace=0.7)
    else:
        grid = plt.GridSpec(4, 1, wspace=0.1, hspace=0.7)

    pos_plot = plt.subplot(grid[0, 0])
    vel_plot = plt.subplot(grid[1, 0], sharex=pos_plot)
    input_plot = plt.subplot(grid[2, 0], sharex=pos_plot)

    if plot_risks and plot_gap:
        risk_plot = plt.subplot(grid[3, 0], sharex=pos_plot)
        gap_plot = plt.subplot(grid[4, 0], sharex=pos_plot)
    elif plot_risks:
        risk_plot = plt.subplot(grid[3, 0], sharex=pos_plot)
    elif plot_gap:
        gap_plot = plt.subplot(grid[3, 0], sharex=pos_plot)

    plot_colors = {TrackSide.LEFT: 'tab:blue',
                   TrackSide.RIGHT: 'tab:orange'}

    # Position plot
    lines = {}
    for side in TrackSide:
        positions = np.array(data['positions'][side])

        if side is TrackSide.LEFT:
            positions[:, 0] = positions[:, 0] - 2.
        else:
            positions[:, 0] = positions[:, 0] + 2.

        lines[side], = pos_plot.plot(positions[:, 1], -positions[:, 0], color=plot_colors[side])

    left_positions_per_second = np.array(data['positions'][TrackSide.LEFT][0::freq])
    left_positions_per_second[:, 0] = left_positions_per_second[:, 0] - 2.

    right_positions_per_second = np.array(data['positions'][TrackSide.RIGHT][0::freq])
    right_positions_per_second[:, 0] = right_positions_per_second[:, 0] + 2.

    for left_point, right_point in zip(left_positions_per_second, right_positions_per_second):
        pos_plot.plot([left_point[1], right_point[1]], [-left_point[0], -right_point[0]], c='lightgrey', linestyle='dashed')
        pos_plot.scatter([left_point[1], right_point[1]], [-left_point[0], -right_point[0]], c='grey')

    y_bounds = (-1.2 * max(max(np.array(data['positions'][TrackSide.LEFT])[:, 0]), max(np.array(data['positions'][TrackSide.RIGHT])[:, 0])) - 2.5,
                -1.2 * min(min(np.array(data['positions'][TrackSide.LEFT])[:, 0]), min(np.array(data['positions'][TrackSide.RIGHT])[:, 0])) + 2.5)

    pos_plot.set_yticks([-12, -2, 2, 12])
    pos_plot.set_yticklabels([10, 0, 0, -10])
    pos_plot.set_ylim(y_bounds)

    pos_plot.set_ylabel('X position [m]')
    pos_plot.legend(lines.values(), lines.keys())
    pos_plot.set_title(title_iterator.__next__(), x=-0.12, y=0.4, va='center')

    # Velocity plot
    for side in TrackSide:
        vel_plot.plot(np.array(data['positions'][side])[:, 1], data['velocities'][side], label=str(side), c=plot_colors[side])

    if mark_replans:
        for side in TrackSide:
            if data['is_replanning'][side]:
                upper_indices = np.array(data['is_replanning'][side]) == 1
                vel_plot.scatter(np.array(data['positions'][side])[upper_indices, 1],
                                 np.array(data['velocities'][side])[upper_indices],
                                 marker='*', c=plot_colors[side])

                lower_indices = np.array(data['is_replanning'][side]) == -1
                vel_plot.scatter(np.array(data['positions'][side])[lower_indices, 1],
                                 np.array(data['velocities'][side])[lower_indices],
                                 marker='o', c=plot_colors[side])
        upper_bound_marker = mpl.lines.Line2D([], [], color='k', marker='*', linestyle='None',  label='Upper bound re-plan')
        lower_bound_marker = mpl.lines.Line2D([], [], color='k', marker='o', linestyle='None', label='Lower bound re-plan')

        vel_plot.legend(handles=[upper_bound_marker, lower_bound_marker], loc='lower right')

    vel_plot.set_title(title_iterator.__next__(), x=-0.12, y=0.4, va='center')
    vel_plot.set_ylabel('Velocity [m/s]')

    # Input (acceleration) plot
    true_acceleration = {}
    for side in TrackSide:
        true_acceleration[side] = np.array(data['accelerations'][side]) - 0.0005 * np.array(data['velocities'][side]) ** 2 - 0.1
        input_plot.plot(np.array(data['positions'][side])[:, 1], true_acceleration[side], label=str(side), c=plot_colors[side])

    y_bounds = (-2.8, 2.8)

    input_plot.set_ylabel('Acceleration [m/s^2]')
    input_plot.set_ylim(y_bounds)
    input_plot.set_title(title_iterator.__next__(), x=-0.12, y=0.4, va='center')

    # risk plot
    if plot_risks:
        bounds_max = max(np.array(data['positions'][TrackSide.LEFT])[-1, 1], np.array(data['positions'][TrackSide.RIGHT])[-1, 1])

        for side in TrackSide:
            risk_plot.plot(np.array(data['positions'][side])[:, 1], data['perceived_risks'][side], c=plot_colors[side])

            if data['risk_bounds'][side] == data['risk_bounds'][side.other]:
                risk_plot.hlines(data['risk_bounds'][side], [0], [bounds_max], linestyles='dashed', colors='grey')
            else:
                risk_plot.hlines(data['risk_bounds'][side], [0], [bounds_max], linestyles='dashed', colors=plot_colors[side])

        risk_plot.set_xlabel('Y position [m]')
        risk_plot.set_ylabel('perceived risk')
        risk_plot.set_ylim((0.0, 1.0))
        risk_plot.set_title(title_iterator.__next__(), x=-0.12, y=0.4, va='center')

    # Gap plot
    if plot_gap:
        gap_data = np.array(data['travelled_distance'][TrackSide.RIGHT]) - np.array(data['travelled_distance'][TrackSide.LEFT]) - 4.8

        gap_plot.plot(np.array(data['positions'][TrackSide.RIGHT])[:, 1], gap_data, c='k')

        gap_plot.set_ylabel('gap [m]')
        gap_plot.set_xlabel('Leading vehicle Y position [m]')
        gap_plot.set_title(title_iterator.__next__(), x=-0.12, y=0.4, va='center')

    return figure


if __name__ == '__main__':
    os.chdir(os.getcwd() + '\\..')

    all_files = ['data/scenario_A.pkl', 'data/scenario_B.pkl', 'data/scenario_C.pkl', 'data/scenario_D.pkl']

    for file in all_files:
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)

        title = os.path.basename(file).replace('_', ' ').replace('.pkl', '').title()
        figure = plot_trial(loaded_data, plot_gap=False, mark_replans=True)
        plt.tight_layout()

        save_file_path = os.path.join('data', 'plots', os.path.basename(file).replace('.pkl', '.png'))

        # figure.savefig(save_file_path, bbox_inches='tight', pad_inches=0.1)

    # show results
    plt.show()
