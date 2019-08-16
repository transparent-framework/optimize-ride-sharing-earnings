"""
This class creates a driver
"""

from __future__ import division
import logging
import numpy as np


class Driver(object):
    """
    Creates a driver
    """

    def __init__(self, config_, driver_id, home_bin, is_strategic):
        """
        Constructor
        :param config_:
        :param driver_id:
        :param home_bin:
        :param is_strategic:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.did = driver_id
        self.home_bin = home_bin
        self.is_strategic = is_strategic
        self.curr_bin = home_bin
        self.next_availability = 0
        self.total_earnings = 0.0

        self.pax_rides = 0
        self.relocation_rides = 0
        self.unsuccessful_waits = 0
        self.coordination_actions = 0

    def get_action(self, q_ind, r_table, t, exploration_factor,
                   allow_coordination, coordination_vector,
                   best_actions):
        """
        Driver makes a decision, either random or independent or coordinated
        :param q_ind:
        :param r_table:
        :param t:
        :param exploration_factor:
        :param allow_coordination:
        :param best_actions:
        :return action:
        """
        if t < self.next_availability:
            action = (0, self.did, -100, -100)
            return action

        # In case previous action was coordination then only allow independent
        # waiting action as next action
        if allow_coordination:
            coordination_factor = coordination_vector[self.curr_bin]
        else:
            coordination_factor = 0.0

        rand1 = np.random.random()
        rand2 = np.random.random()

        if not self.is_strategic:
            coordination = False
            action = q_ind.get_non_strategic_action(self.curr_bin)
        elif rand1 <= coordination_factor:
            # Coordination
            coordination = True
            # Fill up coordinated rebalancing
            action = r_table.get_action(t, self.curr_bin)
        else:
            # Independence
            coordination = False
            if rand2 <= exploration_factor:
                action = q_ind.get_random_action(self.curr_bin)
            else:
                action = best_actions[self.curr_bin]['ind']

        if action == self.curr_bin:
            if coordination:
                action = (1, self.did, self.curr_bin, self.curr_bin)
            else:
                action = (2, self.did, self.curr_bin, self.curr_bin)
        else:
            if coordination:
                action = (3, self.did, self.curr_bin, action)
            else:
                action = (4, self.did, self.curr_bin, action)
        return action

    def update_driver_state(self, t, target, travel_time, earning, cost,
                            coordination, pax_ride, unsuccessful_wait, relocation_ride):
        """
        Update driver object
        :param t:
        :param target:
        :param travel_time:
        :param earning:
        :param cost:
        :param coordination:
        :param pax_ride:
        :param unsuccessful_wait:
        :param relocation_ride:
        """
        if coordination:
            self.coordination_actions += 1
        if pax_ride:
            self.pax_rides += 1
        elif unsuccessful_wait:
            self.unsuccessful_waits += 1
        elif relocation_ride:
            self.relocation_rides += 1

        self.curr_bin = target
        self.total_earnings += (earning - cost)
        self.next_availability = t + travel_time
