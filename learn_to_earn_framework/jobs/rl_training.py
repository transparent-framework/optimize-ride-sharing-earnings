"""
This class implements a job to run the RL training job
"""

import logging
from data.data_exporter import DataExporter
from model_training.run_rl_training import RLTrainer
import pprint


class RunRLTrainingJob(object):
    """
    This class implements a job to run the RL training
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.data_exporter = DataExporter(self.config)

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: RunRLTrainingJob\n")
        self.logger.info("RL training parameters:")

        pp = pprint.PrettyPrinter(indent=4)
        self.logger.info(pp.pprint(self.config['RL_parameters']))

        # Create RL trainer
        rl_trainer = RLTrainer(self.config)

        # Runs RL episodes
        result = rl_trainer.run()
        best_episode = result[0]
        best_model = result[1]
        training_tracker = result[2]

        # Export best_model
        best_model_filename = self.config['RL_parameters'].get('best_model_filename', False)
        best_episode_filename = self.config['RL_parameters'].get('best_episode_filename', False)
        training_tracker_filename = self.config['RL_parameters'].get('training_tracker_filename', False)

        if best_model_filename:
            self.data_exporter.export_model(best_model, best_model_filename)

        if best_episode_filename:
            self.data_exporter.export_episode(best_episode, best_episode_filename)

        if training_tracker_filename:
            self.data_exporter.export_training_tracker(training_tracker, training_tracker_filename)

        self.logger.info("Finished job: RunRLTrainingJob")

        return best_episode
