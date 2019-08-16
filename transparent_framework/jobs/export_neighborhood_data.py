"""
This class implements a job to export the neighborhood data
"""

import logging
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from bin_utils.hex_bin_utils import *

class NeighborhoodDataExportJob(object):
    """
    This class implements a job to export the neighborhood data
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :param viz_name:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.radius = self.config['RL_parameters']['neighborhood_radius']

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: NeighborhoodDataExportJob\n")
        data_provider = DataProvider(self.config)
        data_exporter = DataExporter(self.config)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_bins = hex_attr_df['hex_id'].values

        data = {}
        for r in xrange(self.radius + 1):
            data[r] = {}
            for hex_bin in hex_bins:
                neighbors = hex_neighborhood(hex_bin, hex_attr_df, r)
                zero_vector = np.zeros(len(hex_bins))
                np.put(zero_vector, neighbors, 1)
                one_hot_encoding_vector = zero_vector
                data[r][hex_bin] = one_hot_encoding_vector

        data_exporter.export_neighborhood_data(data)
        self.logger.info("Finished job: NeighborhoodDataExportJob")

