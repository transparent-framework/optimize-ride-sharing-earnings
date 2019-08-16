"""
This class created the independent Q table
"""

from __future__ import division
import logging
import numpy as np
from numba import jit


class Q_independent(object):
    """
    Creates the independent Q table
    """

    def __init__(self, config_, episode_id, hex_bins, T, exploration_factor,
                 neighborhood, popular_bins, hex_distance_df, q_ind=None):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param hex_bins:
        :param T:
        :param exploration_factor:
        :param neighborhood:
        :param popular_bins:
        :param hex_distance_df:
        :param q_ind:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode_id = episode_id
        self.hex_bins = hex_bins
        self.T = T
        self.actions = [action for action in xrange(len(self.hex_bins))]
        RL_parameters = self.config['RL_parameters']
        self.num_episodes = RL_parameters['num_episodes']
        self.exploration_factor = exploration_factor
        self.learning_rate = RL_parameters['learning_rate']
        self.discount_factor = RL_parameters['discount_factor']
        self.objective = RL_parameters['objective']
        self.create_q_table(q_ind)
        self.create_exploration_table(neighborhood)
        self.popular_bins = popular_bins
        self.hex_distance_df = hex_distance_df.pivot(index='pickup_bin', columns='dropoff_bin',
                                                     values='straight_line_distance').fillna(value=0)
        self.hex_distance_matrix = self.hex_distance_df.values

    def create_q_table(self, q_ind):
        """
        Create the actual Q matrix with interface - Q[t][s][a]
        :param q_ind:
        :return:
        """
        if q_ind is not None:
            self.Q_table = q_ind
        else:
            S = len(self.hex_bins)      # Number of hex bins
            A = len(self.actions)       # Number of actions

            # Create independent Q table
            self.Q_table = np.zeros((self.T, S, A), dtype=float)

    def create_exploration_table(self, neighborhood):
        """
        Create exploration probability table
        :param neighborhood:
        :return:
        """
        S = len(self.hex_bins)
        self.exploration_table = np.zeros((S, S), dtype=float)

        for h in self.hex_bins:
            # Gaussian function parameters
            a = 0.7
            c = 1

            max_dist = np.max(neighborhood.keys())
            for dist in range(max_dist, 0, -1):
                prob = a * np.e ** (-1 * (dist * dist) / (2 * c * c))
                neighbors = neighborhood[dist][h] - neighborhood[dist - 1][h]
                num = len(np.nonzero(neighbors)[0])
                for neighbor in np.nonzero(neighbors)[0]:
                    self.exploration_table[h][neighbor] = prob / num

        for h in self.hex_bins:
            self.exploration_table[h][h] = 1 - np.sum(self.exploration_table[h])

    def get_q_value(self, t, s, a):
        """
        Get q value in the q table
        :param t:
        :param s:
        :param a:
        :return q_value:
        """
        if t >= self.T:
            t = self.T - 1

        q_value = self.Q_table[t][s][a]
        return q_value

    def set_q_value(self, t, s, a, value):
        """
        Updates q value in the q table
        :param t:
        :param s:
        :param a:
        :param value:
        :return:
        """
        if t >= self.T:
            t = self.T - 1
        self.Q_table[t][s][a] = value

    @staticmethod
    @jit(nopython=True)
    def jit_get_best_action(t, s, T, q_ind):
        """
        Jit part of the method to find best action
        :param t:
        :param s:
        :param T:
        :param q_ind:
        :return best_action:
        """
        if t >= T:
            t = T - 1

        candidate_actions = q_ind[t][s]
        best_reward = np.max(candidate_actions)
        best_action = (np.random.choice(np.where(candidate_actions == best_reward)[0]))
        return best_action

    def get_best_action(self, t, s):
        """
        Returns best action from the q table
        :param t:
        :param s:
        :return best_action:
        """
        best_action = self.jit_get_best_action(t, s, self.T, self.Q_table)
        return best_action

    def get_random_action(self, s):
        """
        Returns the random action the q table
        :param s:
        :return rand_action:
        """
        rand_action = np.random.multinomial(1, self.exploration_table[s], 1).argmax()
        return rand_action

    def get_non_strategic_action(self, s):
        """
        Returns non-strategic naive driver action
        :param s:
        :return action:
        """
        if np.random.binomial(1, 0.25, 1)[0]:
            target_probabilities = np.take(self.hex_distance_matrix[s], self.popular_bins['hex_id'].values)
            target_probabilities = np.divide(1, target_probabilities, out=np.zeros_like(target_probabilities),
                                             where=target_probabilities != 0)
            target_probabilities /= np.sum(target_probabilities)
            target_idx = np.random.multinomial(1, target_probabilities, 1).argmax()
            return self.popular_bins['hex_id'].values[target_idx]
        else:
            return self.get_random_action(s)

    @staticmethod
    @jit(nopython=True)
    def jit_waiting_q_value(S, t, T, trips, q_ind, travel_time_matrix, reward_matrix,
                            driver_cost_matrix, discount_factor, pickup_objective):
        """
        Jit part of the method to calculate waiting q value
        :param S:
        :param t:
        :param T:
        :param trips:
        :param q_ind:
        :param travel_time_matrix:
        :param reward_matrix:
        :param driver_cost_matrix:
        :param discount_factor:
        :param pickup_objective:
        :return results:
        """
        results = []
        for source in xrange(S):
            A = 0.0
            n = np.sum(trips[source])
            if n == 0:
                continue

            for target in np.nonzero(trips[source])[0]:
                travel_time = travel_time_matrix[source][target]
                earning = reward_matrix[source][target]
                cost = driver_cost_matrix[source][target]
                net_reward = earning - cost
                num_rides = trips[source][target]

                # Matching supply with demand
                if pickup_objective:
                    if source == target:
                        net_reward = num_rides * -1
                    else:
                        net_reward *= 1

                t_prime = t + travel_time
                if t_prime >= T:
                    t_prime = T - 1

                candidate_actions = q_ind[t_prime][target]
                best_reward = np.max(candidate_actions)
                future_best_action = (np.random.choice(np.where(candidate_actions == best_reward)[0]))
                future_q_value = q_ind[t_prime][target][future_best_action]
                A += num_rides * (net_reward + (discount_factor * future_q_value))

            results.append([float(source), float(A), float(n)])
        return results

    def update_waiting_q_value(self, t, trips, city_state):
        """
        Updates q values for waiting actions
        :param t:
        :param trips:
        :param city_state:
        :return:
        """
        S = len(self.hex_bins)
        travel_time_matrix = city_state['travel_time_matrix']
        reward_matrix = city_state['reward_matrix']
        driver_cost_matrix = city_state['driver_cost_matrix']
        q_ind = self.Q_table
        if self.objective == 'pickups':
            pickup_objective = True
        else:
            pickup_objective = False
        results = self.jit_waiting_q_value(
            S, t, self.T, trips, q_ind, travel_time_matrix, reward_matrix,
            driver_cost_matrix, self.discount_factor, pickup_objective)

        for result in results:
            source = int(result[0])
            A = result[1]
            n = int(result[2])
            new_value = ((1 - self.learning_rate)
                         * (self.get_q_value(t, source, source))
                         + (self.learning_rate / n)
                         * (A))
            self.set_q_value(t, source, source, new_value)

    @staticmethod
    @jit(nopython=True)
    def jit_relocate_q_value(S, t, T, trips, q_ind, travel_time_matrix,
                             driver_cost_matrix, discount_factor, pickup_objective):
        """
        Jit part of the method to calculate relocate q value
        :param S:
        :param t:
        :param T:
        :param trips:
        :param q_ind:
        :param travel_time_matrix:
        :param driver_cost_matrix:
        :param discount_factor:
        :param pickup_objective:
        :return results:
        """
        results = []
        for source in xrange(S):
            for target in np.nonzero(trips[source])[0]:
                n = trips[source][target]
                travel_time = travel_time_matrix[source][target]
                cost = driver_cost_matrix[source][target]
                net_reward = -1 * cost * n

                if pickup_objective:
                    net_reward = 0

                t_prime = t + travel_time
                if t_prime >= T:
                    t_prime = T - 1
                candidate_actions = q_ind[t_prime][target]
                best_reward = np.max(candidate_actions)
                future_best_action = (np.random.choice(np.where(candidate_actions == best_reward)[0]))
                future_q_value = q_ind[t_prime][target][future_best_action] * n
                A = net_reward + (discount_factor * future_q_value)

                results.append([float(source), float(target), float(A), float(n)])
        return results

    def update_relocate_q_value(self, t, trips, city_state):
        """
        Updates q values for relocate actions
        :param t:
        :param trips:
        :param city_state:
        :return:
        """
        S = len(self.hex_bins)
        travel_time_matrix = city_state['travel_time_matrix']
        driver_cost_matrix = city_state['driver_cost_matrix']
        q_ind = self.Q_table
        if self.objective == 'pickups':
            pickup_objective = True
        else:
            pickup_objective = False
        results = self.jit_relocate_q_value(
            S,
            t,
            self.T,
            trips,
            q_ind,
            travel_time_matrix,
            driver_cost_matrix,
            self.discount_factor,
            pickup_objective)

        for result in results:
            source = int(result[0])
            target = int(result[1])
            A = result[2]
            n = int(result[3])

            new_value = ((1 - self.learning_rate)
                         * (self.get_q_value(t, source, target))
                         + (self.learning_rate / n)
                         * (A))

            self.set_q_value(t, source, target, new_value)
