"""
This class implements experiment_05
We vary the number of strategic drivers
"""

from __future__ import division
import logging
import numpy as np
from pathos.pools import ProcessPool
import multiprocessing as mp
from copy import deepcopy
from jobs.rl_training import RunRLTrainingJob
from data.data_exporter import DataExporter


class Experiment05(object):
    """
    Experiment05 class
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
        self.expt_name = "expt_05"
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
        num_drivers = self.config['RL_parameters']['num_drivers']
        percent_strategic_drivers = np.arange(0, 1.1, 0.1)
        num_strategic_drivers = [int(x * num_drivers) for x in percent_strategic_drivers]

        # Create a pool of processes
        num_processes = mp.cpu_count()
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0
        for drivers in num_strategic_drivers:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            self.config['RL_parameters']['num_strategic_drivers'] = drivers
            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_05")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_05")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
