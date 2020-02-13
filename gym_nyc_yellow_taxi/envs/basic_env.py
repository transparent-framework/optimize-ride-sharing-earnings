import gym
import logging
import numpy as np
from gym import spaces
from data.data_provider import DataProvider
from ..utils.simulator import initialize_driver_distribution
from ..utils.simulator import take_action


class BasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("gym_logger")
        data_provider = DataProvider(self.config)

        # City state parameters
        self.city_states = data_provider.read_city_states()
        self.hex_attr_df = data_provider.read_hex_bin_attributes()
        self.hex_bins = self.hex_attr_df['hex_id']

        self.T = len(self.city_states)  # Number of time steps
        self.S = len(self.hex_bins)  # Number of hex bins

        # Environment parameters
        self.num_drivers = self.config['env_parameters']['num_drivers']
        self.distribution = self.config['env_parameters']['driver_distribution']
        self.next_free_timestep = np.zeros(self.num_drivers)  # Next free timestep for each driver
        self.total_driver_earnings = np.zeros(self.num_drivers)  # Total earnings for each driver

        # Environment action and observation space
        actions = [7 for i in range(self.S)]
        drivers = [self.num_drivers for i in range(self.S)]
        self.action_space = spaces.MultiDiscrete(actions)
        # self.observation_space = spaces.Tuple((
        #     # spaces.Discrete(self.T),  # Time step
        #     spaces.MultiDiscrete(drivers)  # Driver distribution
        # ))
        self.observation_space = spaces.MultiDiscrete(drivers)

        self.reset()

    def step(self, action):
        """
        Advances the environment by 1 time step
        :param action:
        :return:
        """
        assert self.action_space.contains(action)

        # Take action
        driver_distribution_matrix, reward = take_action(
            self.t, action, self.city_states[self.t], self.hex_attr_df, self.driver_distribution_matrix.copy(),
            self.T)

        # Update time step
        self.t += 1
        self.driver_distribution_matrix = driver_distribution_matrix
        if self.t == self.T:
            self.curr_driver_distribution = self.driver_distribution_matrix[self.T - 1]
            done = True
        else:
            self.curr_driver_distribution = self.driver_distribution_matrix[self.t]
            done = False
        return self._get_obs(), reward, done, {}

    def lookahead(self, action, t, city_state, driver_distribution_matrix):
        """
        Look ahead of what happens when a particular action is taken without actually changing driver
        distributions
        :param action:
        :param t:
        :param city_state:
        :param driver_distribution_matrix:
        :return:
        """
        assert self.action_space.contains(action)

        # Take action
        driver_distribution_matrix, reward = take_action(
            t, action, city_state, self.hex_attr_df, driver_distribution_matrix, self.T)

        # Update time step
        t += 1
        if t == self.T:
            curr_driver_distribution = driver_distribution_matrix[self.T - 1]
            done = True
        else:
            curr_driver_distribution = driver_distribution_matrix[t]
            done = False
        return self._get_obs(curr_driver_distribution), driver_distribution_matrix, reward, done, {}

    def _get_obs(self, curr_driver_distribution=None):
        """
        Returns an observation
        :param curr_driver_distribution:
        :return obs:
        """
        # obs = (self.t, np.array([len(x) for x in self.curr_driver_distribution]))
        if curr_driver_distribution is not None:
            obs = np.array([len(x) for x in curr_driver_distribution])
        else:
            obs = np.array([len(x) for x in self.curr_driver_distribution])
        return obs

    def reset(self):
        """
        Resets the environment for time 0
        :param:
        :return:
        """
        self.t = 0
        self.curr_driver_distribution = initialize_driver_distribution(self.S, self.num_drivers, self.distribution)
        self.driver_distribution_matrix = np.empty((self.T, self.S), dtype=object)
        self.driver_distribution_matrix.fill([])
        self.driver_distribution_matrix[0] = self.curr_driver_distribution

        return self._get_obs()
