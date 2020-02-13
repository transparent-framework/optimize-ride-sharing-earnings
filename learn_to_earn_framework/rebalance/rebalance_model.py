"""
This class implements the min cost rebalancing model
"""

from __future__ import division
import numpy as np
from scipy.optimize import minimize
from numba import jit


def create_rebalancing_graph(t, T, imbalance_threshold, hex_bins, cost_matrix,
                             travel_time_matrix, supply_matrix, demand_matrix, q_ind, objective):
    """
    Creates input for LP solver in form of a graph
    :param t:
    :param T:
    :param imbalance_threshold:
    :param hex_bins:
    :param cost_matrix:
    :param travel_time_matrix:
    :param supply_matrix:
    :param demand_matrix:
    :param q_ind:
    :return (excess bins, deficit bins, edges, rewards associated with edges):
    """
    supply_vector = supply_matrix[t]
    demand_vector = demand_matrix[t]
    imbalance_vector = supply_vector - demand_vector

    # Get graph edges
    edges = set()

    # Get excess bins at time t
    # Each element is tuple of form (hex_bin, t, number of excess drivers)
    E = set()
    for i in hex_bins:
        if imbalance_vector[i] >= imbalance_threshold:
            E.add((i, t, imbalance_vector[i]))

    # Get deficit bins at times t > t0
    # Each element is tuple of form (hex_bin, t, number of deficit drivers)
    D = set()
    for _ in E:
        i = _[0]
        for j in hex_bins:
            t_ij = travel_time_matrix[i][j]
            t_j = t + t_ij
            if t_j >= T:
                continue
            imbalance_j = supply_matrix[t_j][j] - demand_matrix[t_j][j]
            if imbalance_j <= -1 * imbalance_threshold:
                D.add((j, t_j, np.abs(imbalance_j)))
                edges.add((i, j, t_j))

    # Create reward vector
    edges = list(edges)
    E = np.array(list(E))
    D = np.array(list(D))

    reward_vector = []
    for e in edges:
        i = e[0]
        j = e[1]
        t_j = e[2]
        # Travel cost from i -> j and waiting reward at j
        cost = cost_matrix[i][j]
        future_reward = q_ind[t_j][j][j]
        current_reward = q_ind[t][i][i]
        reward_vector.append(future_reward - cost - current_reward)
    return (E, D, edges, np.array(reward_vector))


@jit(nopython=True, parallel=True)
def calc_costs(flow_vector, cost_vector):
    """
    Calculates total rewards achieved by flow vector
    :param flow_vector:
    :param cost_vector:
    :return tot_cost:
    """
    tot_cost = 0.0
    for i in xrange(len(flow_vector)):
        tot_cost += flow_vector[i] * cost_vector[i]
    # tot_cost = np.dot(flow_vector, cost_vector)
    return tot_cost


@jit(nopython=True, parallel=True)
def calc_flows(flow_vector, cost_vector):
    """
    Calculates total rewards achieved by flow vector
    :param flow_vector:
    :param cost_vector:
    :return tot_cost:
    """
    tot_flow = 0.0
    for i in xrange(len(flow_vector)):
        tot_flow -= flow_vector[i]
    return tot_flow


@jit(nopython=True, parallel=True)
def get_tot_outgoing_flow(excess_bins, edges, flow_vector):
    """
    Calculates total outgoing flow from excess bins
    :param excess_bins:
    :param edges:
    :param flow_vector:
    :param outgoing_flow_vector:
    """
    outgoing_flow_vector = []
    for i in excess_bins:
        count = 0
        for idx in xrange(len(edges)):
            if edges[idx][0] == i:
                count += flow_vector[idx]
        outgoing_flow_vector.append(count)
    return np.array(outgoing_flow_vector)


@jit(nopython=True, parallel=True)
def get_tot_incoming_flow(deficit_bins, deficit_times, edges, flow_vector):
    """
    Calculates total incoming flow to deficit bins
    :param deficit_bins:
    :param deficit_times:
    :param edges:
    :param flow_vector:
    :return incoming_flow_vector:
    """
    incoming_flow_vector = []
    for j in xrange(len(deficit_bins)):
        count = 0
        for idx in xrange(len(edges)):
            if (edges[idx][1] == deficit_bins[j]) & (edges[idx][2] == deficit_times[j]):
                count += flow_vector[idx]
        incoming_flow_vector.append(count)
    return np.array(incoming_flow_vector)


def maximize_rewards_heuristic(E, D, edges, reward_vector, t, T, hex_bins):
    """
    Maximizes rewards by rebalancing using heuristic
    (maximize flow along edges sorted by descending order of rewards)
    :param E:
    :param D:
    :param edges:
    :param reward_vector:
    :param t:
    :param T:
    :param hex_bins:
    """
    flow_vector = np.zeros(len(edges), dtype=int)
    excess_vector = np.zeros(len(hex_bins), dtype=int)
    deficit_matrix = np.zeros((T, len(hex_bins)), dtype=int)

    for e in E:
        i = e[0]
        t = e[1]
        excess = e[2]
        excess_vector[i] = excess

    for d in D:
        i = d[0]
        t = d[1]
        deficit = d[2]
        deficit_matrix[t][i] = deficit

    desc_rewards = np.argsort(reward_vector)[::-1]
    for idx in desc_rewards:
        edge = edges[idx]
        i = edge[0]
        j = edge[1]
        t_j = edge[2]
        excess = excess_vector[i]
        deficit = deficit_matrix[t_j][j]
        if excess == 0 or deficit == 0:
            flow = 0
            flow_vector[idx] = flow
        elif excess >= deficit:
            flow = deficit
            excess_vector[i] -= flow
            deficit_matrix[t_j][j] -= flow
            flow_vector[idx] = flow
        else:
            flow = excess
            excess_vector[i] -= flow
            deficit_matrix[t_j][j] -= flow
            flow_vector[idx] = flow

    excess_vector = np.array([x[2] for x in E])
    excess_bins = np.array([x[0] for x in E])
    return (flow_vector, excess_vector, get_tot_outgoing_flow(excess_bins, edges, flow_vector))


def maximize_rewards(self, E, D, edges, reward_vector, t, T, hex_bins):
    """
    Maximizes rewards by rebalancing
    :param E:
    :param D:
    :param edges:
    :param reward_vector:
    :return flow_vector:
    """
    flow_vector = np.zeros(len(edges))
    excess_vector = np.array([x[2] for x in E])
    excess_bins = np.array([x[0] for x in E])

    deficit_vector = np.array([x[2] for x in D])
    deficit_bins = np.array([x[0] for x in D])
    deficit_times = np.array([x[1] for x in D])

    # Turning rewards to costs because standard formulation is minimization of costs
    cost_vector = -1 * reward_vector
    # Constraints
    cons = ({'type': 'ineq',  # Flow >= 0
             'fun': lambda flow_vector: flow_vector},
            {'type': 'ineq',  # Flow outgoing from each node <= excess in that node
             'fun': lambda flow_vector: excess_vector - self.get_tot_outgoing_flow(
                 excess_bins, edges, flow_vector)},
            {'type': 'ineq',  # Flow incoming to each node <= deficit in that node
             'fun': lambda flow_vector: deficit_vector - self.get_tot_incoming_flow(
                 deficit_bins, deficit_times, edges, flow_vector)}
            )

    sol = minimize(fun=self.calc_costs,
                   x0=flow_vector,
                   args=(cost_vector),
                   constraints=cons,
                   options={'ftol': 1})

    min_cost_flow = [np.around(x) for x in sol.x]
    return (min_cost_flow, excess_vector, get_tot_outgoing_flow(excess_bins, edges, min_cost_flow))


def maximize_flows_heuristic(E, D, edges, reward_vector, t, T, hex_bins):
    """
    Maximizes rewards by rebalancing using heuristic
    (maximize flow along edges sorted by descending order of rewards)
    :param E:
    :param D:
    :param edges:
    :param reward_vector:
    :param t:
    :param T:
    :param hex_bins:
    """
    flow_vector = np.zeros(len(edges), dtype=int)
    excess_vector = np.zeros(len(hex_bins), dtype=int)
    deficit_matrix = np.zeros((T, len(hex_bins)), dtype=int)

    for e in E:
        i = e[0]
        t = e[1]
        excess = e[2]
        excess_vector[i] = excess

    for d in D:
        i = d[0]
        t = d[1]
        deficit = d[2]
        deficit_matrix[t][i] = deficit

    asc_rewards = np.argsort(reward_vector)
    for idx in asc_rewards:
        edge = edges[idx]
        i = edge[0]
        j = edge[1]
        t_j = edge[2]
        excess = excess_vector[i]
        deficit = deficit_matrix[t_j][j]
        if excess == 0 or deficit == 0:
            flow = 0
            flow_vector[idx] = flow
        elif excess >= deficit:
            flow = deficit
            excess_vector[i] -= flow
            deficit_matrix[t_j][j] -= flow
            flow_vector[idx] = flow
        else:
            flow = excess
            excess_vector[i] -= flow
            deficit_matrix[t_j][j] -= flow
            flow_vector[idx] = flow

    excess_vector = np.array([x[2] for x in E])
    excess_bins = np.array([x[0] for x in E])
    return (flow_vector, excess_vector, get_tot_outgoing_flow(excess_bins, edges, flow_vector))


def maximize_flows(self, E, D, edges, reward_vector, t, T, hex_bins):
    """
    Maximizes flow by rebalancing
    :param E:
    :param D:
    :param edges:
    :param reward_vector:
    :param t:
    :param T:
    :param hex_bins:
    """
    flow_vector = np.zeros(len(edges))
    excess_vector = np.array([x[2] for x in E])
    excess_bins = np.array([x[0] for x in E])

    deficit_vector = np.array([x[2] for x in D])
    deficit_bins = np.array([x[0] for x in D])
    deficit_times = np.array([x[1] for x in D])

    # Turning rewards to costs because standard formulation is minimization of costs
    cost_vector = -1 * reward_vector
    # Constraints
    cons = ({'type': 'ineq',  # Flow >= 0
             'fun': lambda flow_vector: flow_vector},
            {'type': 'ineq',  # Flow outgoing from each node <= excess in that node
             'fun': lambda flow_vector: excess_vector - self.get_tot_outgoing_flow(
                 excess_bins, edges, flow_vector)},
            {'type': 'ineq',  # Flow incoming to each node <= deficit in that node
             'fun': lambda flow_vector: deficit_vector - self.get_tot_incoming_flow(
                 deficit_bins, deficit_times, edges, flow_vector)}
            )

    sol = minimize(fun=self.calc_flows,
                   x0=flow_vector,
                   args=(cost_vector),
                   constraints=cons,
                   options={'ftol': 1})

    min_cost_flow = [np.around(x) for x in sol.x]
    return (min_cost_flow, excess_vector, get_tot_outgoing_flow(excess_bins, edges, min_cost_flow))


def get_r_matrix(arg):
    """
    Creates r_table row ie. a matrix for each time t
    :param arg:
    :return (t, r_matrix):
    """
    t = arg[0]
    S = arg[1]
    r_matrix = np.zeros((S, S), dtype=float)
    supply_matrix = arg[2]
    demand_matrix = arg[3]
    q_ind = arg[4]
    cost_matrix = arg[5]
    travel_time_matrix = arg[6]
    T = arg[7]
    hex_bins = arg[8]
    imbalance_threshold = arg[9]
    objective = arg[10]

    E, D, edges, reward_vector = create_rebalancing_graph(
        t, T, imbalance_threshold,
        hex_bins, cost_matrix, travel_time_matrix, supply_matrix, demand_matrix, q_ind, objective)

    if (len(E) == 0) and (len(D) == 0):
        return (t, r_matrix)
    elif len(E) == 0:
        return (t, r_matrix)
    elif len(D) == 0:
        # All drivers stay in same place
        outgoing_flow = np.array([0 for x in E])
        excess_vector = np.array([x[2] for x in E])
    else:
        # Solve the min cost flow problem (call heuristic methods if you want
        # very similar results, but in a fraction of time.
        if objective == 'pickups':
            min_cost_flow, excess_vector, outgoing_flow = maximize_flows(
                E, D, edges, reward_vector, t, T, hex_bins)
        else:
            min_cost_flow, excess_vector, outgoing_flow = maximize_rewards(
                E, D, edges, reward_vector, t, T, hex_bins)

        for idx in xrange(len(edges)):
            e = edges[idx]
            i = e[0]
            j = e[1]
            flow = min_cost_flow[idx]
            r_matrix[i][j] = flow

    for idx in xrange(len(E)):
        i = E[idx][0]
        excess = excess_vector[idx]
        out_flow = outgoing_flow[idx]
        if excess - out_flow > 0:
            r_matrix[i][i] = excess - out_flow
        r_matrix[i] /= np.sum(r_matrix[i])

    return (t, r_matrix)
