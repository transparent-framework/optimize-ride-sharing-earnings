"""
This class creates one episode of RL
1. Create drivers
2. Load city states
3. Create independent and coordinated Q tables
4. Run the episode
"""

from __future__ import division
import logging
import numpy as np
from driver.driver import Driver
from tracker import EpisodeTracker
from tracker import DriverTracker
from q_table.q_independent import Q_independent
from rebalance.rebalance_table import R_table
from xi_table.xi_table import XiMatrix
from numba import jit
import operator


class Episode(object):
    """
    Creates an episode of RL
    """

    def __init__(self, config_, episode_id, ind_exploration_factor, hex_attr_df, hex_distance_df,
                 city_states, neighborhood, popular_bins, q_ind=None, r_table=None, xi_matrix=None,
                 model_testing=False):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param ind_exploration_factor:
        :param hex_attr_df:
        :param hex_distance_df:
        :param dist_df:
        :param city_states:
        :param neighborhood:
        :param popular_bins:
        :param q_ind:
        :param r_table:
        :param xi_matrix:
        :param model_testing:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode_id = episode_id
        self.hex_bins = hex_attr_df['hex_id']
        self.city_states = city_states
        self.model_testing = model_testing

        # RL parameters
        RL_parameters = self.config['RL_parameters']
        self.num_drivers = RL_parameters['num_drivers']
        self.num_strategic_drivers = RL_parameters['num_strategic_drivers']
        self.allow_coordination = RL_parameters['allow_coordination']
        self.imbalance_threshold = RL_parameters['imbalance_threshold']
        self.ind_exploration_factor = ind_exploration_factor
        self.learning_rate = RL_parameters['learning_rate']
        self.discount_factor = RL_parameters['discount_factor']
        self.city_states = city_states
        self.neighborhood = neighborhood

        self.num_episodes = RL_parameters['num_episodes']
        self.ind_episodes = RL_parameters['ind_episodes']
        self.reb_episodes = RL_parameters['reb_episodes']

        # Create driver tracker and coordination table
        T = len(self.city_states)
        S = len(self.hex_bins)
        self.DT = DriverTracker(config_, episode_id, self.hex_bins, T)

        # Create drivers
        self.drivers = {}
        strategic_drivers = np.random.choice(xrange(self.num_drivers), self.num_strategic_drivers, replace=False)
        ddm = self.DT.initialize_driver_distribution()
        for hex_bin in xrange(S):
            for did in ddm[hex_bin]:
                if did in strategic_drivers:
                    self.drivers[did] = Driver(self.config, did, hex_bin, True)
                else:
                    self.drivers[did] = Driver(self.config, did, hex_bin, False)

        # Create independent Q table
        self.q_ind = Q_independent(config_, episode_id, self.hex_bins, T, ind_exploration_factor,
                                   neighborhood, popular_bins, hex_distance_df, q_ind)

        # Create R table
        self.r_table = R_table(config_, episode_id, self.hex_bins, T, self.learning_rate, r_table)

        # Create Xi Matrix
        self.xi_matrix = XiMatrix(config_, episode_id, self.hex_bins, T, xi_matrix)

        # Create ride count matrix for the current episodes
        self.ride_count_matrix = np.zeros((T, S, S))
        self.unsuccessful_wait_matrix = np.zeros((T, S))

        # Supply and demand matrix
        self.supply_matrix = np.zeros((T, S), dtype=int)
        self.demand_matrix = np.zeros((T, S), dtype=int)

        # Track episode parameters
        self.episode_tracker = EpisodeTracker(self.config, episode_id, S, T)

    def update_supply_demand_matrices(self, t):
        """
        Updates supply and demand matrix for an episode
        :param t:
        :return:
        """
        supply_vector = self.DT.DDM[t].astype(int)
        demand_vector = self.city_states[t]['pickup_vector'].astype(int)
        self.supply_matrix[t] = supply_vector
        self.demand_matrix[t] = demand_vector

    @staticmethod
    @jit(nopython=True)
    def jit_match_with_a_pax(bins, city_ride_count_matrix, episode_ride_count_matrix):
        """
        Jit version of the method to match with passengers
        :param bins:
        :param city_ride_count_matrix:
        :param episode_ride_count_matrix:
        :return (hex_bin, target_bin):
        """
        for hex_bin in bins:
            rides = city_ride_count_matrix[hex_bin] - episode_ride_count_matrix[hex_bin]
            ride_targets = np.nonzero(rides)[0]
            if len(ride_targets) == 0:
                continue
            else:
                target_bin = np.random.choice(ride_targets)
                return (hex_bin, target_bin)
        return None

    def find_pax_ride(self, t, did, curr_bin, city_state):
        """
        Find a pax ride in the pickup neighborhood
        :param t:
        :param did:
        :param curr_bin:
        :param city_state:
        :return (source, target):
        """
        bins = np.array([curr_bin])
        match_pax_ride = self.jit_match_with_a_pax(bins, city_state['ride_count_matrix'], self.ride_count_matrix[t])

        if match_pax_ride is None:
            return (curr_bin, curr_bin)
        else:
            source = match_pax_ride[0]
            target = match_pax_ride[1]
            return (source, target)

    def execute_wait_action(self, driver_action, t, city_state, coordination):
        """
        Executes wait action
        :param driver_action:
        :param t:
        :param city_state:
        :param coordination:
        :return:
        """
        did = driver_action[1]
        curr_bin = self.drivers[did].curr_bin

        # Find pax in neighborhood of current bin if allowed
        curr_bin, target_bin = self.find_pax_ride(t, did, curr_bin, city_state)

        if target_bin is not curr_bin:
            self.ride_count_matrix[t][curr_bin][target_bin] += 1
            travel_time = city_state['travel_time_matrix'][curr_bin][target_bin]
            earning = city_state['reward_matrix'][curr_bin][target_bin]
            cost = city_state['driver_cost_matrix'][curr_bin][target_bin]
            self.episode_tracker.gross_earnings += (earning - cost)
            self.drivers[did].update_driver_state(t, target_bin, travel_time, earning, cost, coordination,
                                                  True, False, False)
        else:
            self.unsuccessful_wait_matrix[t][curr_bin] += 1
            target_bin = curr_bin
            travel_time = 1
            earning = 0
            cost = 0
            self.drivers[did].update_driver_state(t, target_bin, travel_time, earning, cost, coordination,
                                                  False, True, False)

        self.DT.update_driver_tracker(t, target_bin, travel_time)

        if coordination:
            self.cor_wait_trips[curr_bin][target_bin] += 1
        else:
            self.ind_wait_trips[curr_bin][target_bin] += 1

    def execute_relocate_action(self, driver_action, t, city_state, coordination):
        """
        Executes relocate action
        :param driver_action:
        :param t:
        :param city_state:
        :param coordination:
        :return:
        """
        did = driver_action[1]
        curr_bin = self.drivers[did].curr_bin
        target_bin = driver_action[-1]
        travel_time = city_state['travel_time_matrix'][curr_bin][target_bin]
        earning = 0
        cost = city_state['driver_cost_matrix'][curr_bin][target_bin]
        self.episode_tracker.gross_earnings += (earning - cost)

        self.episode_tracker.relocation_rides += 1
        self.drivers[did].update_driver_state(t, target_bin, travel_time, earning, cost, coordination,
                                              False, False, True)

        self.DT.update_driver_tracker(t, target_bin, travel_time)

        if coordination:
            self.cor_relo_trips[curr_bin][target_bin] += 1
        else:
            self.ind_relo_trips[curr_bin][target_bin] += 1

    def execute_driver_actions(self, t, city_state, driver_actions):
        """
        Executes the driver actions in random order
        :param t:
        :param city_state:
        :param js_vector:
        :param driver_actions:
        :return:
        """
        # Arrange drivers by ascending order of driver earnings
        driver_earnings = {}
        for idx in xrange(len(driver_actions)):
            da_did = driver_actions[idx][1]
            da_earnings = self.drivers[da_did].total_earnings
            driver_earnings[idx] = da_earnings
        idx = [x[0] for x in sorted(driver_earnings.items(), key=operator.itemgetter(1))]

        S = len(self.hex_bins)
        self.cor_wait_trips = np.zeros((S, S), dtype=int)
        self.ind_wait_trips = np.zeros((S, S), dtype=int)
        self.cor_relo_trips = np.zeros((S, S), dtype=int)
        self.ind_relo_trips = np.zeros((S, S), dtype=int)

        # for d_act in driver_actions:
        for x in idx:
            d_act = driver_actions[x]
            act = d_act[0]
            if act == 0:
                pass
            elif act == 1:
                self.execute_wait_action(d_act, t, city_state, True)
            elif act == 2:
                self.execute_wait_action(d_act, t, city_state, False)
            elif act == 3:
                self.execute_relocate_action(d_act, t, city_state, True)
            else:
                self.execute_relocate_action(d_act, t, city_state, False)

    def choose_driver_action(self, did, t, best_actions):
        """
        Chooses best action for a driver
        :param did:
        :param t:
        :param best_actions:
        :return action:
        """
        driver = self.drivers[did]
        action = driver.get_action(
            self.q_ind,
            self.r_table,
            t,
            self.ind_exploration_factor,
            self.allow_coordination,
            self.xi_matrix.get_coordination_vector(t),
            best_actions)
        return action

    def get_best_actions(self, t):
        """
        Creates a dict with key as hex bins and values as a dict with best
        action for both independent and coordinated actions
        :param t:
        :return best_actions:
        """
        best_actions = {}
        for hex_bin in self.hex_bins:
            tmp = {}
            tmp['ind'] = self.q_ind.get_best_action(t, hex_bin)
            best_actions[hex_bin] = tmp
        return best_actions

    def run(self):
        """
        Runs an episode
        :param:
        :return:
        """
        T = len(self.city_states)

        # At every time step
        for t in xrange(T):
            city_state = self.city_states[t]
            best_actions = self.get_best_actions(t)
            driver_actions = []

            # Book-keeping before the driver choose actions
            self.episode_tracker.update_q_ind_action_tracker(t, best_actions)
            self.episode_tracker.update_coordination_tracker(t, self.xi_matrix.get_coordination_vector(t))
            self.episode_tracker.update_rebalancing_action_tracker(t, self.r_table.R_table[t])

            # Each driver chooses an action
            vec_choose_driver_action = np.vectorize(self.choose_driver_action)
            vec_driver_actions = vec_choose_driver_action(xrange(self.num_drivers), t, best_actions)
            driver_actions = np.array(vec_driver_actions).T

            # Execute driver actions
            self.execute_driver_actions(t, city_state, driver_actions)

            # Update Q tables
            if not self.model_testing:
                self.q_ind.update_waiting_q_value(t, self.ind_wait_trips, city_state)
                self.q_ind.update_relocate_q_value(t, self.ind_relo_trips, city_state)

            # Update supply demand matrices
            self.update_supply_demand_matrices(t)

            # Book-keeping during the episode
            supply_vector = self.DT.DDM[t].astype(int)
            self.episode_tracker.update_supply_tracker(t, supply_vector)
            unmet_demand_matrix = city_state['ride_count_matrix'] - self.ride_count_matrix[t]
            unmet_demand_vector = np.sum(unmet_demand_matrix, axis=1)
            self.episode_tracker.unmet_demand += np.sum(unmet_demand_matrix)
            self.episode_tracker.update_unmet_demand_tracker(t, unmet_demand_vector)
            successful_wait_vector = np.sum(self.ride_count_matrix[t], axis=1)
            self.episode_tracker.successful_waits += np.sum(successful_wait_vector)
            self.episode_tracker.update_successful_wait_tracker(t, successful_wait_vector)

            unsuccessful_wait_vector = self.unsuccessful_wait_matrix[t]
            self.episode_tracker.unsuccessful_waits += np.sum(unsuccessful_wait_vector)
            self.episode_tracker.update_unsuccessful_wait_tracker(t, unsuccessful_wait_vector)

        if self.allow_coordination and self.episode_id >= (self.num_episodes - self.reb_episodes):
            if not self.model_testing:
                # Update r table
                self.r_table.update_r_table(self.city_states, self.supply_matrix, self.demand_matrix,
                                            self.q_ind, self.learning_rate)
                # Update xi matrix
                self.xi_matrix.update_xi_matrix(self.supply_matrix, self.demand_matrix)

        # Book-keeping after finishing the episode
        driver_pax_rides_vector = []
        driver_relocations_vector = []
        driver_unsuccessful_waits_vector = []
        driver_earnings_vector = []
        driver_coordinations_vector = []

        for did in xrange(self.num_drivers):
            driver = self.drivers[did]
            driver_pax_rides_vector.append(driver.pax_rides)
            driver_relocations_vector.append(driver.relocation_rides)
            driver_unsuccessful_waits_vector.append(driver.unsuccessful_waits)
            driver_earnings_vector.append([driver.total_earnings, driver.is_strategic, driver.home_bin])
            driver_coordinations_vector.append(driver.coordination_actions)

        self.episode_tracker.update_driver_pax_rides_tracker(driver_pax_rides_vector)
        self.episode_tracker.update_driver_relocations_tracker(driver_relocations_vector)
        self.episode_tracker.update_driver_waits_tracker(driver_unsuccessful_waits_vector)
        self.episode_tracker.update_driver_earnings_tracker(driver_earnings_vector)
        self.episode_tracker.update_driver_coordinations_tracker(driver_coordinations_vector)
        self.episode_tracker.update_passenger_wait_times(self.neighborhood, self.city_states)

        return ({
            'q_ind': self.q_ind.Q_table,
            'r_table': self.r_table.R_table,
            'xi_matrix': self.xi_matrix.xi_matrix,
            'episode_tracker': self.episode_tracker
        })
