"""
This class implements a job for creating city state
"""

import logging
from data.data_exporter import DataExporter
from city_state.create_city_state import CityStateCreator


class CreateCityStateJob(object):
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
        self.logger.info("Starting job: CreateCityStateJob\n")

        city_state_creator = CityStateCreator(self.config)
        city_state = city_state_creator.get_city_states()
        self.logger.info("Exporting city states\n")
        filename = self.config['city_state_creator'].get('filename', 'city_states.dill')
        data_exporter = DataExporter(self.config)
        data_exporter.export_city_state(city_state, filename)

        self.logger.info("Finished job: CreateCityStateJob")
