"""
This class implements experiment_02
We vary the number of drivers
"""

from __future__ import division
import logging
import numpy as np
from pathos.pools import ProcessPool
import multiprocessing as mp
from copy import deepcopy
from jobs.rl_training import RunRLTrainingJob
from data.data_exporter import DataExporter


class Experiment02(object):
    """
    Experiment02 class
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.data_exporter = DataExporter(self.config)
        self.logger = logging.getLogger("cuda_logger")
        self.expt_name = "expt_02"
        self.config['RL_parameters']['experiment'] = self.expt_name

    @staticmethod
    def run_rl_training(config):
        rl_trainer = RunRLTrainingJob(config)
        data = rl_trainer.run()
        return data

    def run(self):
        """
        Run experiment
        """
        num_drivers = np.arange(1000, 6500, 500)
        # Create a pool of processes
        num_processes = mp.cpu_count()
        self.logger.info("Processes: {}".format(num_processes))
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0
        for drivers in num_drivers:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            self.config['RL_parameters']['city_states_filename'] = "city_states.dill"
            self.config['RL_parameters']['num_drivers'] = drivers
            self.config['RL_parameters']['num_strategic_drivers'] = drivers
            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_02")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_02")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
