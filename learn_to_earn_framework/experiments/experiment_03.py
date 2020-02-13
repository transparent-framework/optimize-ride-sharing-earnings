"""
This class implements experiment_03
We vary the number of drivers and optimize pickups vs optimize revenue
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


class Experiment03(object):
    """
    Experiment03 class
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
        self.expt_name = "expt_03"
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
        objectives = ['pickups', 'revenue']
        combinations = list(itertools.product(num_drivers, objectives))

        # Create a pool of processes
        num_processes = mp.cpu_count()
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0
        for comb in combinations:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            self.config['RL_parameters']['city_states_filename'] = "city_states.dill"
            self.config['RL_parameters']['num_drivers'] = comb[0]
            self.config['RL_parameters']['num_strategic_drivers'] = comb[0]
            self.config['RL_parameters']['objective'] = comb[1]
            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_03")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_03")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
