"""
This class tracks the details about the running episode
"""

from __future__ import division
import logging
import numpy as np
import pandas as pd


class EpisodeTracker(object):
    """
    This class tracks the details about the running episode
    """

    def __init__(self, config_, episode_id, S, T):
        """
        Constructor
        :param config_:
        :param episode_id:
        :param S:
        :param T:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")
        self.episode_id = episode_id
        self.T = T
        self.S = S
        self.num_drivers = self.config['RL_parameters']['num_drivers']

        # Create supply tracker
        self.ST = []

        # Create unmet demand tracker
        self.UDT = []

        # Create unsuccessful wait tracker
        self.UWT = []

        # Create successful wait tracker
        self.SWT = []

        # Create driver earnings tracker
        self.DET = []

        # Create driver relocations tracker
        self.DRT = []

        # Create driver pax rides tracker
        self.DPRT = []

        # Create driver waits tracker
        self.DWT = []

        # Create driver coordinations tracker
        self.DCT = []

        # Create rides tracker
        self.RT = []

        # Create best q_ind action tracker
        self.QAT = []

        # Create coordination tracker
        self.XT = []

        # Create rebalancing action tracker
        self.RAT = []

        # Create passenger wait time tracker for unmet demand that could have been fulfilled by neighborhood drivers
        self.PWT = []

        # Scaler trackers
        self.gross_earnings = 0.0
        self.relocation_rides = 0
        self.unmet_demand = 0
        self.successful_waits = 0
        self.unsuccessful_waits = 0

    def update_supply_tracker(self, t, supply_vector):
        """
        Updates supply tracker
        :param t:
        :param supply_vector:
        :return:
        """
        for i in xrange(self.S):
            supply = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'supply': supply_vector[i]
            }
            self.ST.append(supply)

    def update_unmet_demand_tracker(self, t, unmet_demand_vector):
        """
        Updates unmet demand tracker at time t
        :param t:
        :param unmet_demand_vector:
        :return:
        """
        for i in xrange(self.S):
            demand = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'unmet_demand': unmet_demand_vector[i]
            }
            self.UDT.append(demand)

    def update_unsuccessful_wait_tracker(self, t, unsuccessful_wait_vector):
        """
        Updates unsuccessful wait tracker at time t
        :param t:
        :param unsuccessful_wait_vector:
        :return:
        """
        for i in xrange(self.S):
            wait = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'unsuccessful_wait': unsuccessful_wait_vector[i]
            }
            self.UWT.append(wait)

    def update_successful_wait_tracker(self, t, successful_wait_vector):
        """
        Updates successful wait tracker at time t
        :param t:
        :param successful_wait_vector:
        :return:
        """
        for i in xrange(self.S):
            wait = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'successful_wait': successful_wait_vector[i]
            }
            self.SWT.append(wait)

    def update_driver_earnings_tracker(self, driver_earnings_vector):
        """
        Updates earnings of individual driver
        :param driver_earning_vector:
        :return:
        """
        for did in xrange(self.num_drivers):
            earnings = {
                'episode_id': self.episode_id,
                'did': did,
                'earnings': driver_earnings_vector[did][0],
                'is_strategic': driver_earnings_vector[did][1],
                'home_bin': driver_earnings_vector[did][2]
            }
            self.DET.append(earnings)

    def update_driver_relocations_tracker(self, driver_relocations_vector):
        """
        Updates number of relocations of individual drivers
        :param driver_relocations_vector:
        :return:
        """
        for did in xrange(self.num_drivers):
            relocations = {
                'episode_id': self.episode_id,
                'did': did,
                'relocations': driver_relocations_vector[did]
            }
            self.DRT.append(relocations)

    def update_driver_pax_rides_tracker(self, driver_pax_rides_vector):
        """
        Updates number of pax rides of individual drivers
        :param driver_pax_rides_vector:
        :return:
        """
        for did in xrange(self.num_drivers):
            pax_rides = {
                'episode_id': self.episode_id,
                'did': did,
                'pax_rides': driver_pax_rides_vector[did]
            }
            self.DPRT.append(pax_rides)

    def update_driver_waits_tracker(self, driver_waits_vector):
        """
        Updates number of unsuccessful waits of individual drivers
        :param driver_waits_vector:
        :return:
        """
        for did in xrange(self.num_drivers):
            waits = {
                'episode_id': self.episode_id,
                'did': did,
                'waits': driver_waits_vector[did]
            }
            self.DWT.append(waits)

    def update_driver_coordinations_tracker(self, driver_coordinations_vector):
        """
        Updates number of coordination actions of individual drivers
        :param driver_coordinations_vector:
        :return:
        """
        for did in xrange(self.num_drivers):
            coordinations = {
                'episode_id': self.episode_id,
                'did': did,
                'coordinations': driver_coordinations_vector[did]
            }
            self.DCT.append(coordinations)

    def update_ride_tracker(self, did, src, target, t, travel_time, earning, cost, action, coordination):
        """
        Adds a ride to ride tracker
        :param did:
        :param src:
        :param target:
        :param t:
        :param travel_time:
        :param earning:
        :param cost:
        :param action:
        :param coordination:
        :return:
        """
        ride = {
            'episode_id': self.episode_id,
            'src': src,
            'target': target,
            'src_time': t,
            'target_time': t + travel_time,
            'earning': earning - cost,
            'action': action,
            'coordination': coordination
        }
        self.RT.append(ride)

    def update_q_ind_action_tracker(self, t, best_actions):
        """
        Updates best actions tracker
        :param t:
        :param best_actions:
        :return:
        """
        for i in xrange(self.S):
            action = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'best_action': best_actions[i]['ind']
            }
            self.QAT.append(action)

    def update_coordination_tracker(self, t, coordination_vector):
        """
        Updates coordination vector tracker
        :param t:
        :param coordination_vector:
        :return:
        """
        for i in xrange(self.S):
            coordination = {
                'episode_id': self.episode_id,
                'hex_bin': i,
                'time': t,
                'coordination_probability': coordination_vector[i]
            }
            self.XT.append(coordination)

    def update_rebalancing_action_tracker(self, t, rebalancing_table):
        """
        Updates rebalancing action tracker
        :param t:
        :param rebalancing_table:
        :return:
        """
        for i in xrange(self.S):
            rebalance_targets = np.nonzero(rebalancing_table[i])[0]
            for j in rebalance_targets:
                rebalance = {
                    'episode_id': self.episode_id,
                    'hex_bin': i,
                    'action': j,
                    'rebalancing_probability': rebalancing_table[i][j]
                }
                self.RAT.append(rebalance)

    def update_passenger_wait_times(self, neighborhood, city_states):
        """
        Tracks waiting times for unmet demand that could have
        been potentially met by passenger in neighborhood
        Each entry (k ,v) in dictionary indicates that v number of
        unmet demands could be met by drivers within k time units.
        :param neighborhood:
        :param city_states:
        :return:
        """
        df_UDT = pd.DataFrame(self.UDT)
        df_UWT = pd.DataFrame(self.UWT)

        m_udt = pd.pivot_table(df_UDT, values=['unmet_demand'], index='time', columns=['hex_bin']).values
        m_uwt = pd.pivot_table(df_UWT, values=['unsuccessful_wait'], index='time', columns=['hex_bin']).values

        ne = neighborhood[3]
        m_n = []
        for i in range(len(m_udt[0])):
            m_n.append(ne[i])
        m_n = np.array(m_n)

        for t in range(len(m_udt)):
            for h in range(len(m_udt[0])):
                waiting_times = {'time': t, 'hex_bin': h, '5_mins': 0, '10_mins': 0, '15_mins': 0, '20_mins': 0}
                unmet_demand = m_udt[t][h]
                tot = 0
                if m_udt[t][h] > 0:
                    neighbors = np.nonzero(m_n[h])[0]
                    for n in neighbors:
                        if m_udt[t][h] == 0:
                            break
                        tot_free_drivers = int(m_uwt[t][n])
                        for d in range(tot_free_drivers):
                            if m_udt[t][h] == 0:
                                break
                            m_udt[t][h] -= 1
                            if city_states[t]['travel_time_matrix'][n][h] <= 1:
                                waiting_times['5_mins'] += 1
                                tot += 1
                            elif city_states[t]['travel_time_matrix'][n][h] <= 2:
                                waiting_times['10_mins'] += 1
                                tot += 1
                            elif city_states[t]['travel_time_matrix'][n][h] <= 3:
                                waiting_times['15_mins'] += 1
                                tot += 1
                            elif city_states[t]['travel_time_matrix'][n][h] <= 4:
                                waiting_times['20_mins'] += 1
                                tot += 1
                            else:
                                pass
                waiting_times['20_mins+'] = unmet_demand - tot
                self.PWT.append(waiting_times)
