"""
This class implements data exporter for storing various data
"""
import logging
import dill
import os
import pyaml


class DataExporter(object):
    """
    This class implements data exporter for storing various data
    """

    def __init__(self, config_):
        """
        Constructor
        """
        self.logger = logging.getLogger("cuda_logger")
        self.config = config_

    def export_city_state(self, city_states, filename='city_states.dill'):
        """
        Exports a dill file containing city states
        :param city_states:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, 'city_states', filename)
        with open(filepath, 'w') as f:
            dill.dump(city_states, f)
        self.logger.info("Exported city states to {}".format(filepath))

    def export_bin_distances(self, dist_df, filename='hex_distances.csv'):
        """
        Exports a csv file containing the distances between hex bins
        :param dist_df:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        dist_df.to_csv(filepath, sep=',', header=True, index=False)

    def export_rmb_data(self, trips_dist_df, weekday):
        """
        Exports the trips data for regression model of a particular weekday
        :param trips_dist_df:
        :param weekday:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filename = 'rmb_data_' + weekday + '.csv'
        filepath = os.path.join(DATA_DIR, "rmb_data", filename)
        trips_dist_df.to_csv(filepath, sep=',', header=True, index=False)

    def export_neighborhood_data(self, data, filename='neighborhood.dill'):
        """
        Exports a dill file containing neighborhood bins at various radius
        :param data:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w') as f:
            dill.dump(data, f)
        self.logger.info("Exported neighborhood to {}".format(filepath))

    def export_episode(self, data, filename='episode.dill'):
        """
        Exports a dill file containing city states
        :param city_states:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w') as f:
            dill.dump(data, f)
        self.logger.info("Exported experiment data to {}".format(filepath))

    def export_model(self, model, filename='model.dill'):
        """
        Exports a dill file containing the model
        :param model:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, 'models', filename)
        with open(filepath, 'w') as f:
            dill.dump(model, f)
        self.logger.info("Exported model to {}".format(filepath))

    def export_training_tracker(self, training_tracker, filename='training_tracker.dill'):
        """
        Exports a dill file containing training tracker
        :param training_tracker:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, 'training_trackers', filename)
        with open(filepath, 'w') as f:
            dill.dump(training_tracker, f)
        self.logger.info("Exported training tracker to {}".format(filepath))
