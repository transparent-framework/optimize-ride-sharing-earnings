"""
We vary the overlap between independent and rebalancing training
The parameters that change in each run are -
1. ind_episodes
2. reb_episodes
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


class Experiment01(object):
    """
    Experiment01 class
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
        self.expt_name = "expt_01"
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
        ind_percent = np.arange(0., 1.1, 0.1)
        reb_percent = np.arange(0., 1.1, 0.1)
        ind_percent[0] = 0.01
        reb_percent[0] = 0.01
        combinations = list(itertools.product(ind_percent, reb_percent))
        num_episodes = self.config['RL_parameters']['num_episodes']

        # Create a pool of processes
        num_processes = mp.cpu_count()
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0
        for comb in combinations:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            ind_episodes = int(comb[0] * num_episodes)
            reb_episodes = int(comb[1] * num_episodes)
            if (ind_episodes + reb_episodes) < num_episodes:
                self.config['RL_parameters']['ind_episodes'] = ind_episodes
                self.config['RL_parameters']['reb_episodes'] = reb_episodes
                configs.append(deepcopy(self.config))
                count += 1

        self.logger.info("Starting expt_01")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_01")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
