"""
This class runs the baselines
"""

import logging
from baselines.cdqn import cDQN
from baselines.ca2c import cA2C
from baselines.a2c import A2C
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from pathos.pools import ProcessPool
import multiprocessing as mp
mpl = mp.log_to_stderr()
mpl.setLevel(logging.ERROR)


class BaselineDriver(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("baseline_logger")
        self.data_provider = DataProvider(self.config)
        self.data_exporter = DataExporter(self.config)

    @staticmethod
    def run_baseline(baseline_config):
        baseline_name = baseline_config['name']
        baseline_count = baseline_config['count']
        config = baseline_config['config']
        city_states = baseline_config['city_states']
        episode_rewards = []
        if baseline_name == "cDQN":
            baseline = cDQN(config)
            rewards = baseline.run(city_states)
            for _ in range(len(rewards)):
                episode_rewards.append({'agent': 'cDQN',
                                        'episode': _,
                                        'run': baseline_count,
                                        'earnings': rewards[_]})

        if baseline_name == "cA2C":
            baseline = cA2C(config)
            rewards = baseline.run(city_states)
            for _ in range(len(rewards)):
                episode_rewards.append({'agent': 'cA2C',
                                        'episode': _,
                                        'run': baseline_count,
                                        'earnings': rewards[_]})
        if baseline_name == "A2C":
            baseline = A2C(config)
            rewards = baseline.run(city_states)
            for _ in range(len(rewards)):
                episode_rewards.append({'agent': 'A2C',
                                        'episode': _,
                                        'run': baseline_count,
                                        'earnings': rewards[_]})
        return episode_rewards

    def run(self):
        self.logger.info("Starting baselines")
        city_states = self.data_provider.read_city_states()
        baseline_list = self.config['baselines']['baseline_list']

        # Create a pool of processes
        num_processes = mp.cpu_count()
        self.logger.info("Processes: {}".format(num_processes))
        pool = ProcessPool(nodes=num_processes)

        configs = []
        for count in range(10):
            for name in baseline_list:
                configs.append({'name': name,
                                'count': count,
                                'config': self.config,
                                'city_states': city_states})

        results = pool.amap(self.run_baseline, configs).get()
        pool.close()
        pool.join()
        pool.clear()

        episode_rewards = []
        for result in results:
            episode_rewards += result

        self.data_exporter.export_baseline_data(episode_rewards)
        self.logger.info("Finished baselines")
