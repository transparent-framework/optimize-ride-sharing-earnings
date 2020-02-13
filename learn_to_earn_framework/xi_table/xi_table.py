"""
This class creates coordination matrix
"""

from __future__ import division
import logging
import numpy as np


class XiMatrix(object):
    """
    Creates a coordination matrix
    """

    def __init__(self, config_, episode_id, hex_bins, T, xi_matrix=None):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param hex_bins:
        :param T:
        :param xi_matrix:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode_id = episode_id
        self.hex_bins = hex_bins
        self.T = T
        self.S = len(self.hex_bins)
        self.imbalance_threshold = self.config['RL_parameters']['imbalance_threshold']
        self.learning_rate = self.config['RL_parameters']['learning_rate']
        self.create_xi_matrix(xi_matrix)

    def create_xi_matrix(self, xi_matrix):
        """
        Creates xi matrix with interface - xi_matrix[t][s]
        :param xi_matrix:
        :return:
        """
        if xi_matrix is None:
            self.xi_matrix = np.zeros((self.T, self.S), dtype=float)
        else:
            self.xi_matrix = xi_matrix

    def get_coordination_vector(self, t):
        """
        Returns coordination vector for time t
        :param t:
        :return coordination_vector:
        """
        coordination_vector = self.xi_matrix[t]
        return coordination_vector

    def update_xi_matrix(self, supply_matrix, demand_matrix):
        """
        Updates the xi matrix based upon the observed supply demand distribution
        during an episode
        :param supply_matrix:
        :param demand_matrix:
        :return:
        """
        for t in xrange(self.T):
            for i in xrange(self.S):
                supply = supply_matrix[t][i]
                demand = demand_matrix[t][i]

                # If imbalance crosses the threshold
                if supply >= (demand + self.imbalance_threshold):
                    excess_ratio = (supply - demand) / supply
                    self.xi_matrix[t][i] = (
                        self.learning_rate * excess_ratio + (1 - self.learning_rate) * self.xi_matrix[t][i])
                elif supply < demand:
                    if self.xi_matrix[t][i] > 0:
                        deficit_ratio = (demand - supply) / demand
                        self.xi_matrix[t][i] = (self.learning_rate * -1 * deficit_ratio
                                                + (1 - self.learning_rate) * self.xi_matrix[t][i])
