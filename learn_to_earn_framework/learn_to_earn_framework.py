"""
This class implements the driver program for the transparent-framework
project
"""

import logging
import sys
import yaml
from jobs import CreateCityStateJob
from jobs import BuildRegressionModelsJob
from jobs import SparseMatrixFillerJob
from jobs import NeighborhoodDataExportJob
from jobs import RunRLTrainingJob
from jobs import RunRLTestingJob
from jobs import ExperimentDriver


class TransparentFramework(object):
    """
    This class implements the driver program for the transparent-framework
    project.
    """

    def __init__(self, config_):
        self.config = config_

        # Initialize the logging
        if self.config['app']['app_logging_level'] == 'DEBUG':
            logging_level = logging.DEBUG
        elif self.config['app']['app_logging_level'] == 'INFO':
            logging_level = logging.INFO
        else:
            logging_level = logging.INFO

        logging.basicConfig(
            format="LOG: %(asctime)-15s:[%(filename)s]: %(message)s",
            datefmt='%m/%d/%Y %I:%M:%S %p')

        self.logger = logging.getLogger("cuda_logger")
        self.logger.setLevel(logging_level)

    def run(self):
        self.logger.info("Starting app: {}".format(self.config['app']['app_name']))

        # Get job list
        job_list = self.config['jobs']['job_list']

        # Execute jobs
        for job_name in job_list:
            if job_name == "create_city_state":
                job = CreateCityStateJob(self.config)
                job.run()
            if job_name == "build_regression_models":
                job = BuildRegressionModelsJob(self.config)
                job.run()
            if job_name == "fill_sparse_matrices":
                job = SparseMatrixFillerJob(self.config)
                job.run()
            if job_name == "export_neighborhood_data":
                job = NeighborhoodDataExportJob(self.config)
                job.run()
            if job_name == "run_rl_training":
                job = RunRLTrainingJob(self.config)
                job.run()
            if job_name == "run_rl_testing":
                job = RunRLTestingJob(self.config)
                job.run()
            if job_name == "run_experiments":
                job = ExperimentDriver(self.config)
                job.run()

        self.logger.info("Finished")


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = "config/learn-to-earn-framework.yaml"

    with open(config_yaml_path) as f:
        config = yaml.load(f)

    # Start driver
    driver = TransparentFramework(config)
    driver.run()
