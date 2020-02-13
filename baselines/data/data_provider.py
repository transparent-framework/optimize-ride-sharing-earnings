"""
This class implements data provider for reading varios data
"""
import logging
import dill
import pandas as pd
import os
import ast


class DataProvider(object):
    """
    This class implements data provider for reading various data.
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        """
        self.config = config_
        self.logger = logging.getLogger("baseline_logger")

    def read_hex_bin_attributes(self):
        """
        Reads the csv file containing the hex bins attributes
        :return df:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        attr_file = os.path.join(
                DATA_DIR,
                "hex_bins",
                "hex_bin_attributes.csv")
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
        # self.logger.info("Finished reading hex bin attributes file")
        return df

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
        # self.logger.info("Finished reading city states data\n")
        return city_states
