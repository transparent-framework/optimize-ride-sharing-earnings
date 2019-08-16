"""
This class creates city state
"""

from __future__ import division
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from data.data_provider import DataProvider

class CityStateCreator(object):
    """
    Creates city state structure
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :returns:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.start_time = self.config["city_state_creator"]["start_time"]
        self.end_time = self.config["city_state_creator"]["end_time"]
        self.time_slice_duration = (self.config["city_state_creator"]
                    ["time_slice_duration"])
        self.time_unit_duration = (self.config["city_state_creator"]
                    ["time_unit_duration"])
        data_provider = DataProvider(self.config)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_dist_df = data_provider.read_hex_bin_distances()
        self.hex_bins = hex_attr_df['hex_id'].values
        self.hex_dist = hex_dist_df[[
                    'pickup_bin',
                    'dropoff_bin',
                    'straight_line_distance']]

    def get_city_states(self):
        """
        Creates city states from start time to end time
        :param:
        :return:
        """
        city_states = []
        start_time  = self.start_time
        end_time    = self.end_time

        # Create array of time slice values between the start and end time
        business_days = self.config['city_state_creator']['business_days']
        business_hours_start = (self.config['city_state_creator']
                    ['business_hours_start'])
        business_hours_end = (self.config['city_state_creator']
                    ['business_hours_end'])
        index = pd.date_range(
                    start=start_time,
                    end=end_time,
                    freq=str(self.time_unit_duration)+'min')

        # Filter only the required days and hours
        index = index[index.day_name().isin(business_days)]
        index = index[(index.hour >= business_hours_start) &
                      (index.hour <= business_hours_end)]
        time_slice_starts = (index -
                    timedelta(minutes=self.time_slice_duration/2))
        time_slice_ends = (index +
                    timedelta(minutes=self.time_slice_duration/2))

        # Create city states
        city_states = {}
        N = len(index.values)

        for t in xrange(N):
            state = {}
            state['time'] = index[t]
            state['time_slice_start'] = time_slice_starts[t]
            state['time_slice_end'] = time_slice_ends[t]
            state['time_slice_duration'] = self.time_slice_duration
            state['time_unit_duration'] = self.time_unit_duration

            # Logging
            self.logger.info(
                    "Creating state for time: {}\n".format(state['time']))

            # Create state data
            state_data = StateData(
                    start_time=time_slice_starts[t],
                    end_time=time_slice_ends[t],
                    time_slice_duration=self.time_slice_duration,
                    time_unit_duration=self.time_unit_duration,
                    hex_bins=self.hex_bins)

            # Populate state data matrices and vectors

            # Transition and ride count matrices
            state_data.create_transition_matrix()
            state['transition_matrix'] = state_data.transition_matrix
            state['ride_count_matrix'] = state_data.ride_count_matrix

            # Pickup and dropoff vectors
            state_data.create_pickup_vector()
            state_data.create_dropoff_vector()
            state['pickup_vector'] = state_data.pickup_vector
            state['dropoff_vector'] = state_data.dropoff_vector

            # Distance matrix
            state_data.create_distance_matrix()
            state['distance_matrix'] = state_data.distance_matrix

            # Travel time matrix
            state_data.create_travel_time_matrix()
            state['travel_time_matrix'] = state_data.travel_time_matrix

            # Reward matrix
            state_data.create_reward_matrix()
            state['reward_matrix'] = state_data.reward_matrix

            # Straight line distance matrix
            state_data.create_geodesic_matrix(self.hex_dist)
            state['geodesic_matrix'] = state_data.geodesic_matrix

            # Assign city state
            city_states[t] = state

        self.logger.info("Finished creating city states")
        return city_states
