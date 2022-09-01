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
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from trackobjects.trackside import TrackSide

if __name__ == '__main__':
    all_files = glob.glob(os.path.join('..', 'data', 'velocity_*.0.pkl'))

    v = []
    steady_state_gap = []

    for file in all_files:
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)

        if loaded_data['end_state'] != 'Finished':
            print(file)
            print(loaded_data['end_state'])

        v.append(loaded_data['velocities'][TrackSide.LEFT][0])

        gaps = np.array(loaded_data['travelled_distance'][TrackSide.RIGHT]) - np.array(loaded_data['travelled_distance'][TrackSide.LEFT]) - \
               loaded_data['simulation_constants'].vehicle_length

        steady_state = sum(gaps[-21:-1])/20
        steady_state_gap.append(steady_state)

    plt.figure()
    plt.plot(v, steady_state_gap)
    plt.scatter(v, steady_state_gap)
    plt.xlabel('Following Vehicle Velocity [m/s]')
    plt.ylabel('Steady-State Gap [m]')
    plt.show()
