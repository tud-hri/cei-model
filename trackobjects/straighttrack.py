import autograd.numpy as np

from trackobjects import Track
from trackobjects.trackside import TrackSide


class StraightTrack(Track):
    def __init__(self, simulation_constants, track_width=4.):
        self._track_width = track_width
        self._track_length = simulation_constants.track_section_length * 2
        self._section_length = simulation_constants.track_section_length

        self._vehicle_width = simulation_constants.vehicle_width
        self._vehicle_length = simulation_constants.vehicle_length

        self._end_point = np.array([0.0, self._track_length])

        self._way_points = [np.array([0.0, 0.0]), self._end_point]

        self._left_start_point = np.array([0.0, 0.0])
        self._right_start_point = np.array([0.0, simulation_constants.track_start_point_distance])

    def is_beyond_track_bounds(self, position):
        _, distance_to_track = self.closest_point_on_route(position)
        return distance_to_track > self.track_width / 2.0

    def is_beyond_finish(self, position):
        return position[1] >= self._end_point[1]

    @staticmethod
    def get_heading(*args):
        return np.pi / 2

    @staticmethod
    def closest_point_on_route(position):
        closest_point_on_route = np.array([.0, position[1]])
        shortest_distance = abs(position[0])

        return closest_point_on_route, shortest_distance

    @staticmethod
    def traveled_distance_to_coordinates(distance, **kwargs):
        return np.array([0.0, distance])

    @staticmethod
    def coordinates_to_traveled_distance(point, **kwargs):
        return point[1]

    def get_collision_bounds_approximation(self, traveled_distance_vehicle_1):
        return self.get_collision_bounds(traveled_distance_vehicle_1, self._vehicle_width, self._vehicle_length, )

    @staticmethod
    def get_collision_bounds(traveled_distance_vehicle_1, vehicle_width, vehicle_length, **kwargs):
        return traveled_distance_vehicle_1 - vehicle_length, traveled_distance_vehicle_1 + vehicle_length

    def get_track_bounding_rect(self) -> (float, float, float, float):
        x1 = - 2 * self.track_width
        x2 = 2 * self.track_width

        y1 = 0.
        y2 = self._track_length

        return x1, y1, x2, y2

    def get_way_points(self, track_side: TrackSide, show_run_up=False) -> list:
        return self._way_points

    def get_start_position(self, track_side: TrackSide) -> np.ndarray:
        if track_side is TrackSide.LEFT:
            return self._left_start_point
        elif track_side is TrackSide.RIGHT:
            return self._right_start_point
        else:
            return None

    @property
    def total_distance(self) -> float:
        return self._track_length

    @property
    def track_width(self) -> float:
        return self._track_width
