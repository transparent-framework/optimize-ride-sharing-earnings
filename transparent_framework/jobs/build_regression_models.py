"""
This class runs the regression model builder job
"""

import logging
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from regression_models.build_regression_models import RegressionModelBuilder
import calendar


class BuildRegressionModelsJob(object):
    """
    This class runs the regression model builder job
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
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: BuildRegressionModelsJob\n")

        # Export the regression models data
        for model in self.config['regression_models']:
            year = model['year'][-2:]
            month = model['month'].lower()
            weekday = list(calendar.day_name).index(model['weekday'])
            (self.logger.info("Creating regression model data {}-{}-{}s"
                    .format(year, month, model['weekday'])))

            rmb = RegressionModelBuilder(self.config,
                                         year,
                                         month,
                                         weekday)
            dist_df = rmb.get_bins_distance_dataframe()
            trips_df = rmb.create_trips_data_with_distance()

            (self.logger.info("Exporting regression model data {}-{}-{}s\n"
                    .format(year, month, model['weekday'])))
            data_exporter = DataExporter(self.config)
            data_exporter.export_bin_distances(dist_df)
            data_exporter.export_rmb_data(
                    trips_df,
                    model['weekday'] + '_' + month + '_' + year)

        self.logger.info("Finished job: BuildRegressionModelsJob")
