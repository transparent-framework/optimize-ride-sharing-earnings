"""
This class fills up sparse city state matrices
"""

from __future__ import division
import logging
import numpy as np
from data.data_provider import DataProvider


class SparseMatrixFiller(object):
    """
    Fills up sparse city state matrices
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        data_provider = DataProvider(self.config)
        filename = self.config['city_state_creator'].get('filename', 'city_states.dill')
        self.city_states = data_provider.read_city_states(filename)
        self.reg_models = data_provider.read_regression_models()

    def fill_matrices(self):
        """
        Fills up sparse matrices
        :param:
        :return:
        """
        self.logger.info("Starting filling up sparse city states")

        for state in self.city_states.values():
            state['reward_matrix'] = self.fill_rewards_matrix(state)
            state['distance_matrix'] = self.fill_distance_matrix(state)
            state['travel_time_matrix'] = self.fill_travel_time_matrix(
                state).astype(int)
            state['driver_cost_matrix'] = self.create_driver_cost_matrix(state)

        self.logger.info("Finished filling up sparse city states")
        return self.city_states

    def fill_rewards_matrix(self, state):
        """
        Fills missing entries in rewards matrix
        :param state:
        :return reward_matrix:
        """
        day = state['time_slice_start'].strftime("%A")
        month = state['time_slice_start'].strftime("%B").lower()
        year = state['time_slice_start'].strftime("%y")
        model = "{}_{}_{}_{}.csv".format(day, "september", year, "fare")
        df = self.reg_models[model]

        hour = state['time_slice_start'].hour
        hour_var = "{}.pickup_hour".format(hour)
        hour_coef = df[df['variable'] == hour_var]['estimate'].values[0]
        dist_coef = (df[df['variable'] == 'straight_line_distance']['estimate']
                     .values[0])
        est_reward_matrix = (state['geodesic_matrix'] * dist_coef) + hour_coef
        new_reward_matrix = np.where(
                    state['reward_matrix'] <= 0,
                    est_reward_matrix,
                    state['reward_matrix'])
        np.fill_diagonal(new_reward_matrix, 0)
        return new_reward_matrix

    def fill_distance_matrix(self, state):
        """
        Fills missing entries in distance matrix
        :param state:
        :return distance_matrix:
        """
        day = state['time_slice_start'].strftime("%A")
        month = state['time_slice_start'].strftime("%B").lower()
        year = state['time_slice_start'].strftime("%y")
        model = "{}_{}_{}_{}.csv".format(day, "september", year, "trip")
        df = self.reg_models[model]

        hour = state['time_slice_start'].hour
        hour_var = "{}.pickup_hour".format(hour)
        hour_coef = df[df['variable'] == hour_var]['estimate'].values[0]
        dist_coef = (df[df['variable'] == 'straight_line_distance']['estimate']
                     .values[0])
        est_dist_matrix = (state['geodesic_matrix'] * dist_coef) + hour_coef
        new_dist_matrix = np.where(
                    state['distance_matrix'] <= 0,
                    est_dist_matrix,
                    state['distance_matrix'])
        np.fill_diagonal(new_dist_matrix, 0)
        return new_dist_matrix

    def fill_travel_time_matrix(self, state):
        """
        Fills missing entries in travel time matrix
        :param state:
        :return travel_time_matrix:
        """
        day = state['time_slice_start'].strftime("%A")
        month = state['time_slice_start'].strftime("%B").lower()
        year = state['time_slice_start'].strftime("%y")
        model = "{}_{}_{}_{}.csv".format(day, "september", year, "duration")
        df = self.reg_models[model]

        hour = state['time_slice_start'].hour
        hour_var = "{}.pickup_hour".format(hour)
        hour_coef = df[df['variable'] == hour_var]['estimate'].values[0]
        dist_coef = (df[df['variable'] == 'straight_line_distance']['estimate']
                     .values[0])
        est_travel_matrix = (state['geodesic_matrix'] * dist_coef) + hour_coef
        est_travel_matrix = est_travel_matrix/60
        est_travel_matrix = np.ceil(
                    est_travel_matrix/state['time_unit_duration'])
        new_travel_matrix = np.where(
                    state['travel_time_matrix'] <= 0,
                    est_travel_matrix,
                    state['travel_time_matrix'])
        np.fill_diagonal(new_travel_matrix, 0)
        return new_travel_matrix

    def create_driver_cost_matrix(self, state):
        """
        Creates a matrix containing driver costs
        :param state:
        :return driver_cost_matrix:
        """
        mile_fare = 0.57
        driver_cost_matrix = mile_fare * state['distance_matrix']
        return driver_cost_matrix
