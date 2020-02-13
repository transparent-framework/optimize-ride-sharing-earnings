"""
These methods help match passengers to drivers during each step
"""
import numpy as np


def initialize_driver_distribution(S, num_drivers, distribution='uniform'):
    """
    Creates driver distribution at time 0
    :param S:
    :param num_drivers:
    :param distribution:
    :return driver_distribution:
    """
    drivers = np.arange(num_drivers)
    driver_distribution = np.array_split(drivers, S)
    return [x.tolist() for x in driver_distribution]


def get_neighbor(hex_attr_df, hex_bin, direction):
    """
    Get neighbor
    :param hex_attr_df:
    :param hex_bin:
    :param direction:
    :return neighbor:
    """
    attributes = hex_attr_df.iloc[hex_bin]
    try:
        neighbor = int(attributes[direction])
    except ValueError:
        neighbor = None
    return neighbor


def take_relocate_action(t, T, drivers, action, hex_bin, hex_attr_df, city_state, driver_distribution_matrix):
    """
    Takes relocate action for all drivers in the current hex_bin
    :param t:
    :param T:
    :param drivers:
    :param action:
    :param hex_bin:
    :param hex_attr_df:
    :param city_state:
    :param driver_distribution_matrix:
    :return driver_distribution_matrix, reward:
    """
    reward = 0
    act_dict = {0: "north_east_neighbor",
                1: "north_neighbor",
                2: "north_west_neighbor",
                3: "south_east_neighbor",
                4: "south_neighbor",
                5: "south_west_neighbor"
                }
    neighbor = get_neighbor(hex_attr_df, hex_bin, act_dict[action[hex_bin]])

    # If the relocate action is infeasible because the hex bin is on the map boundary
    if neighbor is None:
        travel_time = 1
        neighbor = hex_bin
        reward -= 1000
    else:
        travel_time = int(city_state['travel_time_matrix'][hex_bin][neighbor])
        if travel_time == 0:  # travel_time is at least 1 (this condition is for sanity)
            travel_time = 1
        reward -= len(drivers) * city_state['driver_cost_matrix'][hex_bin][neighbor]

    t_prime = t + travel_time
    if t_prime < T:
        driver_distribution_matrix[t_prime][neighbor] = driver_distribution_matrix[t_prime][neighbor] + drivers

    return driver_distribution_matrix, reward


def take_wait_action(t, T, drivers, action, hex_bin, city_state, driver_distribution_matrix):
    """
    Takes wait action for all drivers in the current hex_bin
    :param t:
    :patam T:
    :param drivers:
    :param action:
    :param hex_bin:
    :param city_state:
    :param driver_distribution_matrix:
    :return driver_distribution_matrix, reward:
    """
    reward = 0
    pax_destination_vector = city_state['ride_count_matrix'][hex_bin].astype(int)
    # Assign drivers to passengers
    driver_destination_vector = np.split(drivers, np.cumsum(pax_destination_vector))
    # Drivers who did not find passengers are the last element of vector
    # Assign them to appropriate bin
    driver_destination_vector[hex_bin] = driver_destination_vector[-1]
    driver_destination_vector = driver_destination_vector[:-1]

    # Update next driver distribution and update total rewards
    for i in range(len(action)):
        if len(driver_destination_vector[i]) > 0:
            if i != hex_bin:  # If the driver was matched with a passenger
                travel_time = int(city_state['travel_time_matrix'][hex_bin][i])
                if travel_time == 0:
                    travel_time = 1
                reward += (
                    len(driver_destination_vector[i])
                    * (city_state['driver_cost_matrix'][hex_bin][i] - city_state['reward_matrix'][hex_bin][i])
                )
            else:  # If the driver waited unsuccessfully
                travel_time = 1
                reward += 0

            t_prime = t + travel_time
            if t_prime < T:
                driver_distribution_matrix[t_prime][i] = (
                    driver_distribution_matrix[t_prime][i] + driver_destination_vector[i].tolist()
                )

    return driver_distribution_matrix, reward


def take_action(t, action, city_state, hex_attr_df, driver_distribution_matrix, T):
    """
    Takes chosen action
    :param t:
    :param action:
    :param city_state:
    :param hex_attr_df:
    :param driver_distribution_matrix:
    :param T:
    :return driver_distribution_matrix, reward:
    """
    rewards = np.zeros(len(action), dtype=float)

    for hex_bin in range(len(action)):
        # Each hex_bin has a separate action associated with it
        drivers = list(driver_distribution_matrix[t][hex_bin])

        if len(drivers) == 0:
            continue

        if action[hex_bin] <= 5:       # relocate to neighbor bin
            driver_distribution_matrix, reward = take_relocate_action(t, T, drivers, action, hex_bin,
                                                                      hex_attr_df, city_state,
                                                                      driver_distribution_matrix)
            rewards[hex_bin] = reward

        else:  # action[hex_bin] == 6  # wait action
            driver_distribution_matrix, reward = take_wait_action(t, T, drivers, action, hex_bin, city_state,
                                                                  driver_distribution_matrix)
            rewards[hex_bin] = reward

    return driver_distribution_matrix, rewards
