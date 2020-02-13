"""
This class implements experiment_04
We vary the imbalance threshold and optimize revenue
"""

from __future__ import division
import logging
import numpy as np
import itertools
from pathos.pools import ProcessPool
import multiprocessing as mp
from copy import deepcopy
from jobs.rl_training import RunRLTrainingJob
from data.data_exporter import DataExporter
mpl = mp.log_to_stderr()
mpl.setLevel(logging.ERROR)


class Experiment04(object):
    """
    Experiment04 class
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
        self.expt_name = "expt_04"
        self.config['RL_parameters']['experiment'] = self.expt_name

    @staticmethod
    def run_rl_training(config):
        rl_trainer = RunRLTrainingJob(config)
        try:
            data = rl_trainer.run()
        except BaseException:
            print config
            raise ValueError
        return data

    def run(self):
        """
        Run experiment
        """
        num_drivers = np.arange(1000, 6500, 500)
        thresholds = np.arange(5, 55, 5)
        thresholds = np.insert(thresholds, 0, 2)
        combinations = list(itertools.product(num_drivers, thresholds))

        # Create a pool of processes
        num_processes = mp.cpu_count()
        self.logger.info("Processes: {}".format(num_processes))
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0
        for comb in combinations:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            self.config['RL_parameters']['num_drivers'] = comb[0]
            self.config['RL_parameters']['imbalance_threshold'] = comb[1]
            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_04")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_04")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
