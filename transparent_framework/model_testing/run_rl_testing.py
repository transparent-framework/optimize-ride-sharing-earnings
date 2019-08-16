"""
This class runs the model testing
"""

from __future__ import division
import logging
from data.data_provider import DataProvider
from episode.episode import Episode
from tracker import TrainingTracker  # Can be used without any changes as testing tracker


class RLTester(object):
    """
    Creates RL testing object
    """

    def __init__(self, config_, grid_search=False):
        """
        Constructor
        :param config_:
        :param grid_search:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.expt_name = self.config['Model_testing']['experiment']
        self.test_parameters = self.config['Model_testing']

        # Create testing tracker
        self.testing_tracker = TrainingTracker(self.config)

    def run(self):
        """
        Creates and runs training episode
        :param:
        :return:
        """
        data_provider = DataProvider(self.config)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_distance_df = data_provider.read_hex_bin_distances()
        city_states = data_provider.read_city_states(self.test_parameters['city_states_filename'])
        model = data_provider.read_model(self.test_parameters['model_filename'])
        neighborhood = data_provider.read_neighborhood_data()
        popular_bins = data_provider.read_popular_hex_bins()

        q_ind = model['q_ind']
        r_table = model['r_table']
        xi_matrix = model['xi_matrix']

        episode_id = 0

        # Create episode
        ind_exploration_factor = 0.0

        episode = Episode(self.config,
                          episode_id,
                          ind_exploration_factor,
                          hex_attr_df,
                          hex_distance_df,
                          city_states,
                          neighborhood,
                          popular_bins,
                          q_ind,
                          r_table,
                          xi_matrix,
                          True)

        # Run episode
        tables = episode.run()
        q_ind = tables['q_ind']
        r_table = tables['r_table']
        xi_matrix = tables['xi_matrix']
        episode_tracker = tables['episode_tracker']

        self.testing_tracker.update_RL_tracker(
            0, episode_tracker.gross_earnings,
            episode_tracker.successful_waits, episode_tracker.unsuccessful_waits,
            episode_tracker.unmet_demand, episode_tracker.relocation_rides,
            episode_tracker.DET, episode_tracker.DPRT, episode_tracker.DWT,
            episode_tracker.DRT, episode_tracker.DCT)

        self.logger.info("""
                         Expt: {} Earnings: {}
                         Model: {}
                         Test day: {}
                         Num drivers: {}
                         Pax rides: {} Relocation rides: {} Unmet demand: {}
                         """.format(self.expt_name, episode_tracker.gross_earnings,
                                    self.test_parameters['model_filename'],
                                    self.test_parameters['city_states_filename'],
                                    self.config['RL_parameters']['num_drivers'],
                                    episode_tracker.successful_waits,
                                    episode_tracker.relocation_rides,
                                    episode_tracker.unmet_demand))
        self.logger.info("----------------------------------")

        return self.testing_tracker
