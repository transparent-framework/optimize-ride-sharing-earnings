"""
This class tracks the details about the running RL training
"""

from __future__ import division
import logging
import pandas as pd
import numpy as np
import scipy.stats


class TrainingTracker(object):
    """
    This class tracks the details about the running RL training
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :param S:
        :param T:
        :return:
        """
        self.config = config_
        self.logger = logging.getLogger("cuda_logger")

        # Create RL training tracker
        self.RLT = []

    def mean_confidence_interval(self, data, confidence=0.95):
        """
        Calculates mean and confidence interval around the data array
        :param data:
        :param confidence:
        :return m, m-h, m+h
        """
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        if confidence == 0:
            # Return 1 standard error interval on both sides of the mean
            return m, m-se, m+se
        else:
            return m, m-h, m+h

    def update_RL_tracker(self, episode_id, gross_earnings, gross_successful_driver_waits,
                          gross_unsuccessful_driver_waits, gross_unmet_demand, gross_relocation_rides,
                          driver_earnings_tracker, driver_pax_rides_tracker, driver_waits_tracker,
                          driver_relocations_tracker, driver_coordinations_tracker):
        """
        Updates RL training tracker
        :param episode_id:
        :param gross_earnings:
        :param gross_successful_driver_waits:
        :param gross_unsuccessful_driver_waits:
        :param gross_unmet_demand:
        :param gross_relocation_rides:
        :param driver_earnings_tracker:
        :param driver_pax_rides_tracker:
        :param driver_waits_tracker:
        :param driver_relocations_tracker:
        :param driver_coordinations_tracker:
        :return:
        """
        df_det = pd.DataFrame(driver_earnings_tracker)
        earnings = self.mean_confidence_interval(df_det['earnings'].values, confidence=0)
        earnings_mean = earnings[0]
        earnings_lower_se = earnings[1]
        earnings_upper_se = earnings[2]

        df_dprt = pd.DataFrame(driver_pax_rides_tracker)
        pax_rides = self.mean_confidence_interval(df_dprt['pax_rides'].values, confidence=0)
        pax_rides_mean = pax_rides[0]
        pax_rides_lower_se = pax_rides[1]
        pax_rides_upper_se = pax_rides[2]

        df_dwt = pd.DataFrame(driver_waits_tracker)
        waits = self.mean_confidence_interval(df_dwt['waits'].values, confidence=0)
        waits_mean = waits[0]
        waits_lower_se = waits[1]
        waits_upper_se = waits[2]

        df_drt = pd.DataFrame(driver_relocations_tracker)
        relocations = self.mean_confidence_interval(df_drt['relocations'].values, confidence=0)
        relocations_mean = relocations[0]
        relocations_lower_se = relocations[1]
        relocations_upper_se = relocations[2]

        df_dct = pd.DataFrame(driver_coordinations_tracker)
        coordinations = self.mean_confidence_interval(df_dct['coordinations'].values, confidence=0)
        coordinations_mean = coordinations[0]
        coordinations_lower_se = coordinations[1]
        coordinations_upper_se = coordinations[2]

        rl_episode = {
            'episode_id': episode_id,
            'gross_earnings': gross_earnings,
            'gross_successful_driver_waits': gross_successful_driver_waits,
            'gross_unsuccessful_driver_waits': gross_unsuccessful_driver_waits,
            'gross_unmet_demand': gross_unmet_demand,
            'gross_relocation_rides': gross_relocation_rides,
            'earnings_mean': earnings_mean,
            'earnings_lower_se_bound': earnings_lower_se,
            'earnings_upper_se_bound': earnings_upper_se,
            'successful_waits_mean': pax_rides_mean,
            'successful_waits_lower_se_bound': pax_rides_lower_se,
            'successful_waits_upper_se_bound': pax_rides_upper_se,
            'unsuccessful_waits_mean': waits_mean,
            'unsuccessful_waits_lower_se_bound': waits_lower_se,
            'unsuccessful_waits_upper_se_bound': waits_upper_se,
            'relocations_mean': relocations_mean,
            'relocations_lower_se_bound': relocations_lower_se,
            'relocations_upper_se_bound': relocations_upper_se,
            'coordinations_mean': coordinations_mean,
            'coordinations_lower_se_bound': coordinations_lower_se,
            'coordinations_upper_se_bound': coordinations_upper_se
        }
        self.RLT.append(rl_episode)
