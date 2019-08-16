"""
This class creates the rebalance table
"""

from __future__ import division
import logging
import numpy as np
from rebalance.rebalance_model import get_r_matrix
from multiprocessing import Pool


class R_table(object):
    """
    Creates the rebalancing table
    """

    def __init__(self, config_, episode_id, hex_bins, T, learning_rate, R_table=None):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param hex_bins:
        :param T:
        :param R_table:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode_id = episode_id
        self.hex_bins = hex_bins
        self.T = T
        self.learning_rate = learning_rate
        self.create_R_table(R_table)
        self.imbalance_threshold = self.config['RL_parameters']['imbalance_threshold']
        self.objective = self.config['RL_parameters']['objective']

    def create_R_table(self, R_table):
        """
        Creates the actual rebalancing table with interface R[t][s][a]
        Each entry contains the probability of taking action a at time t
        at state s
        :param R_table:
        :return:
        """
        if R_table is not None:
            self.R_table = R_table
        else:
            S = len(self.hex_bins)
            # Create rebalance table
            self.R_table = np.zeros((self.T, S, S), dtype=float)

    def get_action(self, t, hex_bin):
        """
        Returns an action based upon the probabilities stored in
        the R_table
        :param t:
        :param hex_bin:
        :return action:
        """
        rebalance_destination_vector = self.R_table[t][hex_bin]
        try:
            action = np.random.choice(self.hex_bins, p=rebalance_destination_vector)
        except ValueError:
            action = hex_bin
        return action

    def create_parallel_args(self, supply_matrix, demand_matrix, q_ind, city_states):
        """
        Creates args for pool of processes
        :param supply_matrix:
        :param demand_matrix:
        :param q_ind:
        :param city_states:
        :return args:
        """
        args = {}
        for t in xrange(self.T):
            args[t] = {
                    'city_state': city_states[t],
                    'supply_matrix': supply_matrix,
                    'demand_matrix': demand_matrix,
                    'q_ind': q_ind
                    }
        return args

    def update_r_table(self, city_states, supply_matrix, demand_matrix, q_ind, learning_rate):
        """
        Updates R table for coordination
        :param city_states:
        :param supply_matrix:
        :param demand_matrix:
        :param q_ind:
        :param learning_rate:
        :return:
        """
        S = len(self.hex_bins)
        self.new_R_table = np.zeros((self.T, S, S), dtype=float)

        args = []
        for t in range(self.T - 1, 0, -1):
            args.append([t, S, supply_matrix, demand_matrix, q_ind.Q_table, city_states[t]['driver_cost_matrix'],
                         city_states[t]['travel_time_matrix'], self.T, self.hex_bins, self.imbalance_threshold,
                         self.objective])

        pool = Pool(24)
        results = pool.map_async(get_r_matrix, args, chunksize=96).get()
        pool.close()
        pool.join()

        for res in results:
            t = res[0]
            r_matrix = res[1]
            if type(t) is not int:
                self.logger.info("t: {}".format(t))
                self.logger.info("r_matrix: {}".format(r_matrix))
            self.new_R_table[t] = r_matrix

        if np.all(self.R_table == 0):
            self.R_table = self.new_R_table
        else:
            self.R_table = ((1 - learning_rate) * self.R_table) + (learning_rate * self.new_R_table)
