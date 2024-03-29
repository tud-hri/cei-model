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
import abc

import numpy as np
from trackobjects.trackside import TrackSide


class Track(abc.ABC):
    @abc.abstractmethod
    def is_beyond_track_bounds(self, position: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def is_beyond_finish(self, position: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def get_heading(self, position: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def closest_point_on_route(self, position: np.ndarray) -> (np.ndarray, float):
        pass

    @abc.abstractmethod
    def traveled_distance_to_coordinates(self, distance: float, track_side: TrackSide) -> np.ndarray:
        pass

    @abc.abstractmethod
    def coordinates_to_traveled_distance(self, point: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def get_collision_bounds_approximation(self, traveled_distance_vehicle_1: float) -> (float, float):
        pass

    @abc.abstractmethod
    def get_collision_bounds(self, traveled_distance_vehicle_1: float, vehicle_width: float, vehicle_length: float) -> (float, float):
        pass

    @abc.abstractmethod
    def get_track_bounding_rect(self) -> (float, float, float, float):
        pass

    @abc.abstractmethod
    def get_way_points(self, track_side: TrackSide, show_run_up=False) -> list:
        pass

    @abc.abstractmethod
    def get_start_position(self, track_side: TrackSide) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def total_distance(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def track_width(self) -> float:
        pass
