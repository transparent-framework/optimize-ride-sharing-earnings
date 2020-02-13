"""
This class implements the driver program for the baselines
"""

import logging
import sys
import yaml
from jobs.run_baselines import BaselineDriver


class Baselines(object):
    """
    This class implements the driver program for the baselines
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

        self.logger = logging.getLogger("baseline_logger")
        self.logger.setLevel(logging_level)

    def run(self):
        self.logger.info("Starting app: {}".format(self.config['app']['app_name']))

        # Get job list
        job_list = self.config['jobs']['job_list']

        # Execute jobs
        for job_name in job_list:
            if job_name == "run_baselines":
                job = BaselineDriver(self.config)
                job.run()

        self.logger.info("Finished")


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = "config/baseline_config.yaml"

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Baselines(config)
    driver.run()
