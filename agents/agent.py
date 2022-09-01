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


class Agent(abc.ABC):
    @abc.abstractmethod
    def compute_discrete_input(self, dt):
        """
        This method should calculate a discrete acceleration input; either -1, 0 or 1.

        :param dt: the length of the time step (needed for some forms of MPC)
        :return: one of (-1, 0, 1)
        """
        pass

    @abc.abstractmethod
    def compute_continuous_input(self, dt):
        """
        This method should calculate an acceleration input; expressed between -1 and 1.

        :param dt: the length of the time step (needed for some forms of MPC)
        :return: acceleration between -1 and 1
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        This method should reset the agent object so it can start a new data collection run

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass
