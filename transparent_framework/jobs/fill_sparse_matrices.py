"""
This class implements a job to fill sparse city state matrices
"""

import logging
from data.data_exporter import DataExporter
from sparse_matrices.fill_sparse_matrices import SparseMatrixFiller


class SparseMatrixFillerJob(object):
    """
    This class implements a job for creating city state
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: SparseMatrixFillerJob\n")

        sparse_matrix_filler = SparseMatrixFiller(self.config)
        city_state = sparse_matrix_filler.fill_matrices()
        self.logger.info("Exporting city states\n")
        filename = self.config['city_state_creator'].get('filename', 'city_states.dill')
        data_exporter = DataExporter(self.config)
        data_exporter.export_city_state(city_state, filename)

        self.logger.info("Finished job: SparseMatrixFillerJob")
