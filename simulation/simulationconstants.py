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


class SimulationConstants:
    """ object that stores all constants needed to recall a saved simulation. """

    def __init__(self, dt, vehicle_width, vehicle_length, track_start_point_distance, track_section_length, max_time):
        self.dt = dt
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.track_start_point_distance = track_start_point_distance
        self.track_section_length = track_section_length
        self.max_time = max_time
