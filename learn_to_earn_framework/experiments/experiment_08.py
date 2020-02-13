"""
This class implements experiment_08
We use the saved models from experiment_07 and test for generalizations conditional to changes in demand,
number of drivers and imbalance threshold from training day to test day
"""

from __future__ import division
import os
import logging
from pathos.pools import ProcessPool
import multiprocessing as mp
from copy import deepcopy
from jobs.rl_testing import RunRLTestingJob
from data.data_exporter import DataExporter


class Experiment08(object):
    """
    Experiment08 class
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
        self.expt_name = "expt_08"
        self.config['Model_testing']['experiment'] = self.expt_name

    @staticmethod
    def run_rl_testing(config):
        rl_tester = RunRLTestingJob(config)
        data = rl_tester.run()
        return data

    def run(self):
        """
        Run experiment
        """
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        weeks_of_month = ['00', '01', '02', '03', '04']
        imbalance_thresholds = [2]
        model_num_drivers = [4000, 5000, 6000, 7000, 8000, 9000, 10000]

        test_combinations = []
        for model_day in days:
            for model_wom in weeks_of_month:
                for model_threshold in imbalance_thresholds:
                    for model_drivers in model_num_drivers:
                        model_args = [model_day, model_wom, str(model_drivers), str(model_threshold)]
                        model_filename = "_".join(model_args) + "_model.dill"
                        if os.path.isfile(self.config['app']['DATA_DIR'] + 'models/' + model_filename):
                            for test_wom in weeks_of_month:
                                for test_drivers in range(model_drivers-3000, model_drivers+4000, 1000):
                                    test_file = model_day + '_' + test_wom + '_city_states.dill'
                                    if os.path.isfile(self.config['app']['DATA_DIR'] + 'city_states/' + test_file):
                                        test_combinations.append({'model': model_filename,
                                                                  'test_dow': model_day,
                                                                  'test_wom': test_wom,
                                                                  'test_drivers': test_drivers})
        self.logger.info("Total test combinations: {}".format(len(test_combinations)))

        # Create a pool of processes
        num_processes = mp.cpu_count()
        pool = ProcessPool(nodes=num_processes)

        configs = []
        count = 0

        for comb in test_combinations:
            self.config['Model_testing']['experiment'] = self.expt_name + "_" + str(count)
            self.config['Model_testing']['city_states_filename'] = (
                comb['test_dow'] + '_' + comb['test_wom'] + '_city_states.dill')
            self.config['Model_testing']['model_filename'] = comb['model']
            self.config['RL_parameters']['num_drivers'] = comb['test_drivers']
            self.config['RL_parameters']['num_strategic_drivers'] = comb['test_drivers']

            configs.append(deepcopy(self.config))
            count += 1

        self.logger.info("Starting expt_08")

        results = pool.amap(self.run_rl_testing, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        self.logger.info("Finished expt_08")

        # Export best episode
        self.data_exporter.export_episode(results, self.expt_name + ".dill")
