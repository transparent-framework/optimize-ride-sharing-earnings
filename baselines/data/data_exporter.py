"""
This class implements data exporter for storing various data
"""
import logging
import dill
import os


class DataExporter(object):
    """
    This class implements data exporter for storing various data
    """

    def __init__(self, config_):
        """
        Constructor
        """
        self.logger = logging.getLogger("baseline_logger")
        self.config = config_

    def export_baseline_data(self, data, filename='baselines.dill'):
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            dill.dump(data, f)
        self.logger.info("Exported baseline data to {}".format(filepath))
