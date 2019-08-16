"""
This class builds regression models for predicting sparse features
"""
from __future__ import division
import logging
from data.data_provider import DataProvider
from geopy.distance import geodesic
import itertools
import pandas as pd
import math


class RegressionModelBuilder(object):
    """
    This class builds regression models for predicting sparse features
    """
    def __init__(self, config_, year, month, weekday):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.year = year
        self.month = month
        self.weekday = weekday
        self.logger = logging.getLogger("cuda_logger")
        data_provider = DataProvider(self.config)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_attr_df['center'] = hex_attr_df.apply(
                self.calculate_bin_center,
                axis=1)
        self.hex_attr_df = hex_attr_df

    @staticmethod
    def calculate_bin_center(s):
        """
        Calculates coordinates of the center of hex bin
        :param s:
        :return [center_longitude, center_latitude]:
        """
        s_west_lon = float(s['west'][0])
        s_west_lat = float(s['west'][1])
        s_east_lon = float(s['east'][0])
        s_east_lat = float(s['east'][1])
        s_cent_lon = (s_west_lon + s_east_lon) / 2
        s_cent_lat = (s_west_lat + s_east_lat) / 2
        return [s_cent_lon, s_cent_lat]

    @staticmethod
    def calculate_distance_miles(s):
        """
        Calculates distance in miles between a pair of geocoordinates
        :param s:
        :return dist:
        """
        dist = geodesic(s['pickup_center'], s['dropoff_center']).miles
        return dist

    @staticmethod
    def calculate_compass_bearing(s):
        """
        Calculates compass bearing between a pair of geocoordinates
        :param s:
        :return compass_bearing:
        """
        pickup_lat  = float(s['pickup_center'][1])
        pickup_lon  = float(s['pickup_center'][0])
        dropoff_lat = float(s['dropoff_center'][1])
        dropoff_lon = float(s['dropoff_center'][0])

        lat1 = math.radians(pickup_lat)
        lat2 = math.radians(dropoff_lat)

        diff_lon = math.radians(dropoff_lon - pickup_lon)

        x = math.sin(diff_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) *
                math.cos(diff_lon))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def get_bins_distance_dataframe(self):
        """
        Creates a dataframe for all combinations of bin distances in miles
        :param:
        :return df:
        """
        pair_iters = itertools.permutations(self.hex_attr_df['hex_id'], 2)
        pairs = [pair for pair in pair_iters]
        pairs_df = pd.DataFrame(pairs, columns=['pickup_bin', 'dropoff_bin'])
        hex_df = self.hex_attr_df[['hex_id', 'center']]

        # Add pickup and dropoff bin coordinates

        df = pd.merge(
                pairs_df,
                hex_df,
                left_on='pickup_bin',
                right_on='hex_id',
                how='inner')
        df = df[['pickup_bin', 'dropoff_bin', 'center']]
        df.columns = ['pickup_bin', 'dropoff_bin', 'pickup_center']

        df = pd.merge(
                df,
                hex_df,
                left_on='dropoff_bin',
                right_on='hex_id',
                how='inner')

        df = df[['pickup_bin', 'dropoff_bin', 'pickup_center', 'center']]
        df.columns = [
                'pickup_bin',
                'dropoff_bin',
                'pickup_center',
                'dropoff_center']

        # Calculate straightline distance
        df['straight_line_distance'] = df.apply(
                self.calculate_distance_miles,
                axis=1)

        # Calculate compass bearing
        df['compass_bearing'] = df.apply(
                self.calculate_compass_bearing,
                axis=1)

        return df

    def get_trips_data(self):
        """
        Gets data for appropriate weekday and month
        :param:
        :return df:
        """
        # Create SQL query
        query = """ \
                SELECT \
                    *,
                    HOUR(tpep_pickup_datetime) as pickup_hour\
                FROM \
                    `yt-{0}-{1}` \
                WHERE \
                    WEEKDAY(tpep_pickup_datetime) like {2} \
                AND pickup_bin IS NOT NULL \
                AND dropoff_bin IS NOT NULL \
                AND pickup_bin != dropoff_bin; \
                """

        # Read query output and select required columns
        df = DataProvider.read_sql_query(query.format(self.month,
                                                      self.year,
                                                      self.weekday))
        cols = ['tpep_pickup_datetime',
                'tpep_dropoff_datetime',
                'trip_distance',
                'fare_amount',
                'duration_seconds',
                'pickup_bin',
                'dropoff_bin',
                'pickup_hour']
        df = df[cols]

        return df

    def create_trips_data_with_distance(self):
        """
        Merges trips data with distances df
        :param:
        :return trips_dist_df:
        """
        self.logger.info("Querying trips data from database")
        trips_df = self.get_trips_data()
        self.logger.info("Creating straight line distance matrix")
        dist_df = self.get_bins_distance_dataframe()
        trips_dist_df = pd.merge(
                trips_df,
                dist_df,
                on=['pickup_bin','dropoff_bin'],
                how='inner')

        return trips_dist_df
