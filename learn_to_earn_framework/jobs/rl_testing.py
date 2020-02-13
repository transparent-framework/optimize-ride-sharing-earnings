"""
This class implements a job to run the RL testing job
"""

import logging
from data.data_exporter import DataExporter
from model_testing.run_rl_testing import RLTester
import pprint


class RunRLTestingJob(object):
    """
    This class implements a job to run the RL testing
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.data_exporter = DataExporter(self.config)

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: RunRLTestingJob\n")
        self.logger.info("RL testing parameters:")

        # Create RL trainer
        rl_tester = RLTester(self.config)

        # Runs RL episodes
        result = rl_tester.run()

        self.logger.info("Finished job: RunRLTestingJob")
        return result
