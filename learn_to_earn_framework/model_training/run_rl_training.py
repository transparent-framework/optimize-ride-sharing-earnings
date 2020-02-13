"""
This class runs the RL Training
"""

from __future__ import division
import logging
import numpy as np
from data.data_provider import DataProvider
from episode.episode import Episode
from tracker import TrainingTracker
from tqdm import tqdm


class RLTrainer(object):
    """
    Creates RL training object
    """

    def __init__(self, config_, grid_search=False):
        """
        Constructor
        :param config_:
        :param grid_search:
        :return:
        """
        self.config = config_
        self.grid_search = grid_search
        self.logger = logging.getLogger("cuda_logger")
        self.expt_name = self.config['RL_parameters']['experiment']
        self.objective = self.config['RL_parameters']['objective']
        self.city_states_filename = self.config['RL_parameters']['city_states_filename']

        # Create training tracker
        self.training_tracker = TrainingTracker(self.config)

    def run(self):
        """
        Creates and runs training episode
        :param:
        :return:
        """
        data_provider = DataProvider(self.config)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_distance_df = data_provider.read_hex_bin_distances()
        city_states = data_provider.read_city_states(self.city_states_filename)
        neighborhood = data_provider.read_neighborhood_data()
        popular_bins = data_provider.read_popular_hex_bins()
        num_episodes = self.config['RL_parameters']['num_episodes']
        ind_episodes = self.config['RL_parameters']['ind_episodes']
        exp_decay_multiplier = self.config['RL_parameters']['exp_decay_multiplier']

        q_ind = None
        r_table = None
        xi_matrix = None

        best_episode = None
        best_model = {}

        progress_bar = tqdm(xrange(num_episodes))
        for episode_id in progress_bar:
            progress_bar.set_description("Episode: {}".format(episode_id))
            current_best = -1000000

            # Create episode
            ind_exploration_factor = np.e ** (-1 * episode_id * exp_decay_multiplier / ind_episodes)

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
                              xi_matrix)

            # Run episode
            tables = episode.run()
            q_ind = tables['q_ind']
            r_table = tables['r_table']
            xi_matrix = tables['xi_matrix']
            episode_tracker = tables['episode_tracker']

            # Uncomment for logging if running a job, comment during experiments
            # otherwise it leads to insanely huge logging output which is useless

            # self.logger.info("""
            #                  Expt: {} Episode: {} Earnings: {}
            #                  Pax rides: {} Relocation rides: {} Unmet demand: {}
            #                  """.format(self.expt_name, episode_id,
            #                             episode_tracker.gross_earnings,
            #                             episode_tracker.successful_waits,
            #                             episode_tracker.relocation_rides,
            #                             episode_tracker.unmet_demand))
            # self.logger.info("----------------------------------")

            self.training_tracker.update_RL_tracker(
                episode_id, episode_tracker.gross_earnings,
                episode_tracker.successful_waits, episode_tracker.unsuccessful_waits,
                episode_tracker.unmet_demand, episode_tracker.relocation_rides,
                episode_tracker.DET, episode_tracker.DPRT, episode_tracker.DWT,
                episode_tracker.DRT, episode_tracker.DCT)

            # Keep track of the best episode
            if self.objective == 'revenue':
                if episode_tracker.gross_earnings >= current_best:
                    best_episode = episode_tracker
                    current_best = best_episode.gross_earnings
            else:  # self.objective == 'pickups':
                if episode_tracker.successful_waits >= current_best:
                    best_episode = episode_tracker
                    current_best = episode_tracker.successful_waits

            # Keep track of the best model
            best_model['ind_exploration_factor'] = ind_exploration_factor
            best_model['config'] = self.config
            best_model['q_ind'] = q_ind
            best_model['r_table'] = r_table
            best_model['xi_matrix'] = xi_matrix
            best_model['training_tracker'] = self.training_tracker

        # After finishing training
        self.logger.info("Expt: {} Earnings: {} Met Demand: {} Unmet Demand: {}".format(self.expt_name,
                                                                         best_episode.gross_earnings,
                                                                         best_episode.successful_waits,
                                                                         best_episode.unmet_demand))
        return best_episode, best_model, self.training_tracker
