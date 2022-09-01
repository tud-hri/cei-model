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
