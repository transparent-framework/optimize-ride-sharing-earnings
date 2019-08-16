"""
This class implements data provider for reading varios data
"""
import logging
import dill
import pandas as pd
from dbutils import get_db_connection
import os
import ast
import json


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
        self.logger = logging.getLogger("cuda_logger")

    def read_geojson_file(self):
        """
        Reads the geojson file containing hex bins
        :returns dict:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        geojson_file = os.path.join(
                DATA_DIR,
                "hex_bins",
                "nyc_hex_bins.geojson")
        with open(geojson_file, 'r') as f:
            js = json.load(f)
        return js

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

    def read_hex_bin_distances(self):
        """
        Reads the csv file containing the hex bin distances
        :return df:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        dist_file = os.path.join(
                DATA_DIR,
                "hex_bins",
                "hex_distances.csv")
        df = pd.read_csv(
                dist_file,
                header=0,
                index_col=False)
        return df

    @classmethod
    def read_sql_query(cls, query):
        """
        Reads the output of a sql query
        :param query:
        :return df:
        """
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
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

    def read_model(self, filename='model.dill'):
        """
        Reads the saved RL model
        :param filename:
        :return model:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        model_file = os.path.join(DATA_DIR, 'models', filename)
        with open(model_file, 'rb') as f:
            model = dill.load(f)
        return model

    def read_regression_models(self):
        """
        Reads the regression models built in Stata
        :param:
        :return rmb_model:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        rmb_dir = os.path.join(DATA_DIR, "rmb_models")
        rmb_files = os.listdir(rmb_dir)
        rmb_model = {}
        for model in rmb_files:
            f = os.path.join(rmb_dir, model)
            data = pd.read_csv(f, header=0, index_col=False)
            data.columns = ['variable', 'estimate']
            rmb_model[model] = data
        self.logger.info("Finished reading regression models")
        return rmb_model

    def read_neighborhood_data(self, filename='neighborhood.dill'):
        """
        Reads the neighborhood data dill file
        :param filename:
        :return neighborhood:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        neighborhood_file = os.path.join(DATA_DIR, filename)
        with open(neighborhood_file, 'rb') as f:
            neighborhood = dill.load(f)
        # self.logger.info("Finished reading neighborhood data")
        return neighborhood

    def read_popular_hex_bins(self):
        """
        Reads the csv file containing the popular hex bins
        for naive drivers
        :return df:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        pop_bin_file = os.path.join(
                    DATA_DIR,
                    "hex_bins",
                    "popular_hex_bins.csv")
        df = pd.read_csv(
                pop_bin_file,
                header=0,
                index_col=False)
        return df
