"""
This class implements experiment_06
We create and save one model for first occurance of each day of the week in month of September
"""

from __future__ import division
import logging
from pathos.pools import ProcessPool
import multiprocessing as mp
from copy import deepcopy
from jobs.rl_training import RunRLTrainingJob
from data.data_exporter import DataExporter


class Experiment06(object):
    """
    Experiment06 class
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
        self.expt_name = "expt_06"
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
        days = [
            'Sunday_00_', 'Monday_00_', 'Tuesday_00_', 'Wednesday_00_', 'Thursday_00_', 'Friday_00_', 'Saturday_00_',
            'Sunday_01_', 'Monday_01_', 'Tuesday_01_', 'Wednesday_01_', 'Thursday_01_', 'Friday_01_', 'Saturday_01_',
            'Sunday_02_', 'Monday_02_', 'Tuesday_02_', 'Wednesday_02_', 'Thursday_02_', 'Friday_02_', 'Saturday_02_',
            'Sunday_03_', 'Monday_03_', 'Tuesday_03_', 'Wednesday_03_', 'Thursday_03_', 'Friday_03_', 'Saturday_03_'
        ]

        # Create a pool of processes
        num_processes = mp.cpu_count()
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0

        for day in days:
            self.config['RL_parameters']['experiment'] = self.expt_name + "_" + str(count)
            self.config['RL_parameters']['city_states_filename'] = day + 'city_states.dill'
            self.config['RL_parameters']['best_model_filename'] = day + 'model.dill'
            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_06")

        results = pool.amap(self.run_rl_training, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_06")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
