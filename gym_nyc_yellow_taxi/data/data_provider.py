"""
This class implements data provider for reading various data
"""
import dill
import os
import pandas as pd
import ast
import json


class DataProvider(object):
    """
    This class implements data provider for reading various data
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_

    def read_city_states(self, filename='city_states.dill'):
        """
        Reads the city states dill file
        :param filename:
        :return city_states:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        city_states_file = os.path.join(DATA_DIR, 'city_states', filename)
        with open(city_states_file, 'rb') as f:
            city_states = dill.load(f)
        return city_states

    def read_geojson_file(self):
        """
        Reads the geojson file containing hex bins
        :param:
        :return dict:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        geojson_file = os.path.join(DATA_DIR, "hex_bins", "nyc_hex_bins.geojson")
        with open(geojson_file, 'r') as f:
            js = json.load(f)
        return js

    def read_hex_bin_attributes(self):
        """
        Reads the csv file containing the hex bin attributes
        :param:
        :return df:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        attr_file = os.path.join(DATA_DIR, "hex_bins", "hex_bin_attributes.csv")
        df = pd.read_csv(
                attr_file,
                header=0,
                index_col=False,
                converters={'east': ast.literal_eval,
                            'north_east': ast.literal_eval,
                            'north_west': ast.literal_eval,
                            'south_east': ast.literal_eval,
                            'south_west': ast.literal_eval,
                            'west': ast.literal_eval})
        return df
