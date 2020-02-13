"""
This class implements experiment_00
It simply runs a RL training session, exports model, best episode and training tracker
"""

from __future__ import division
import logging
from jobs.rl_training import RunRLTrainingJob


class Experiment00(object):
    """
    Experiment00 class
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
        """
        Run experiment
        """
        # Coordination experiment
        self.expt_name = "expt_00"
        self.config['RL_parameters']['experiment'] = self.expt_name
        self.config['RL_parameters']['city_states_filename'] = "city_states.dill"
        self.config['RL_parameters']['best_model_filename'] = "expt_00_model.dill"
        self.config['RL_parameters']['best_episode_filename'] = "expt_00_episode.dill"
        self.config['RL_parameters']['training_tracker_filename'] = "expt_00_training_tracker.dill"
        self.config['RL_parameters']['allow_coordination'] = True

        # Hyperparameter values
        self.config['RL_parameters']['ind_episodes'] = 60
        self.config['RL_parameters']['reb_episodes'] = 160
        self.config['RL_parameters']['discount_factor'] = 0.99

        self.logger.info("Starting expt_00 coordination")

        rl_trainer = RunRLTrainingJob(self.config)
        rl_trainer.run()

        self.logger.info("Finished expt_00 coordination")

        # # No coordination experiment
        # self.expt_name = "expt_00_no_coordination"
        # self.config['RL_parameters']['experiment'] = self.expt_name
        # self.config['RL_parameters']['city_states_filename'] = "city_states.dill"
        # self.config['RL_parameters']['best_model_filename'] = "expt_00_no_coordination_model.dill"
        # self.config['RL_parameters']['best_episode_filename'] = "expt_00_no_coordination_episode.dill"
        # self.config['RL_parameters']['training_tracker_filename'] = "expt_00_no_coordination_training_tracker.dill"
        # self.config['RL_parameters']['allow_coordination'] = False

        # self.logger.info("Starting expt_00 no coordination")

        # rl_trainer = RunRLTrainingJob(self.config)
        # rl_trainer.run()

        # self.logger.info("Finished expt_00 no coordination")
