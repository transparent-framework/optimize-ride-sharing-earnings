"""
This class tracks the driver trips and distribution in an episode
"""

from __future__ import division
import logging
import numpy as np


class DriverTracker(object):
    """
    Tracks driver trips and distribution
    """

    def __init__(self, config_, episode_id, hex_bins, T):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param hex_bins:
        :param T:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode = episode_id
        self.hex_bins = hex_bins
        self.T = T

        # Create driver distribution matrix for the episode
        self.DDM = self.create_driver_distribution_matrix()

    def create_driver_distribution_matrix(self):
        """
        Driver distribution across hex bins
        Each entry of matrix is an array of driver ids
        :param:
        :return DDM:
        """
        S = len(self.hex_bins)          # Number of hex bins
        DDM = np.zeros((self.T, S), dtype=int)
        return DDM

    def initialize_driver_distribution(self):
        """
        Create driver distribution at time 0
        :param:
        :return ddm:
        """
        dist = self.config['RL_parameters']['driver_distribution']
        num_drivers = self.config['RL_parameters']['num_drivers']
        drivers = [i for i in xrange(num_drivers)]
        S = len(self.hex_bins)
        if dist == 'uniform':
            ddm = np.array_split(drivers, S)
        else:
            ddm = np.array_split(drivers, S)
        for i in xrange(len(self.hex_bins)):
            self.DDM[0][i] = len(ddm[i])
        return ddm

    def update_driver_tracker(self, t, target, travel_time):
        """
        Updates driver tracker object for a given driver
        :param t:
        :param target:
        :param travel_time:
        :return:
        """
        # Update driver distribution
        update_t = t + travel_time
        if update_t >= self.T:
            update_t = self.T - 1

        self.DDM[update_t][target] += 1
