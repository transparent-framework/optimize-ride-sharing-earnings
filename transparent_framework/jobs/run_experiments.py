"""
This class run the experiments
"""

from __future__ import division
import logging
from experiments import Experiment00
from experiments import Experiment01
from experiments import Experiment02
from experiments import Experiment03
from experiments import Experiment04
from experiments import Experiment05
from experiments import Experiment06
from experiments import Experiment07
from experiments import Experiment08


class ExperimentDriver(object):
    """
    Creates experiment driver
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
        self.logger.info("Starting experiments")

        # Get experiment list
        expt_list = self.config['experiments']['expt_list']

        # Run experiments
        for expt_name in expt_list:
            if expt_name == "experiment_00":
                expt = Experiment00(self.config)
                expt.run()
            if expt_name == "experiment_01":
                expt = Experiment01(self.config)
                expt.run()
            if expt_name == "experiment_02":
                expt = Experiment02(self.config)
                expt.run()
            if expt_name == "experiment_03":
                expt = Experiment03(self.config)
                expt.run()
            if expt_name == "experiment_04":
                expt = Experiment04(self.config)
                expt.run()
            if expt_name == "experiment_05":
                expt = Experiment05(self.config)
                expt.run()
            if expt_name == "experiment_06":
                expt = Experiment06(self.config)
                expt.run()
            if expt_name == "experiment_07":
                expt = Experiment07(self.config)
                expt.run()
            if expt_name == "experiment_08":
                expt = Experiment08(self.config)
                expt.run()

        self.logger.info("Finished experiments")
