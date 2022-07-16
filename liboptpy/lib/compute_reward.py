import numpy as np
from liboptpy.simulator.bioprocess import bioprocess_simulator
from liboptpy.lib.util import comupute_cumulative_feed
import pandas as pd
from tqdm import tqdm

def compute_reward(cumulative_feed, state, horizon):
    titer = state[1]
    total_CA = state[1] * state[-1]
    oil_consumed = cumulative_feed * 917 - state[2] * state[-1]
    yield_CA = total_CA / oil_consumed
    pv = titer / horizon
    man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
    total_reward = - man_cost * horizon  # * state[-1] * state[1]
    return total_reward

def compute_avg_reward(bn, params, policy, time_measurement,  real_measurement, B, isMDP=True):
    reward = np.zeros(B)
    for b in tqdm(range(B)):
        x0 = bn.initial_state_full + np.abs(
            np.random.normal(0, np.array(bn.initial_state_full) / 10 + 0.01))
        # time_measurement = [4 * i for i in range(bn.n_time)]
        # time_measurement[-1] = 135.9
        simulator = bioprocess_simulator(bn, time_measurement, policy, real_measurement)
        states = simulator.g(time_measurement, x0, params)
        feed = pd.DataFrame(simulator.feed_every_second, columns=['time', 'feed_rate'],
                               dtype="float64").drop_duplicates().to_numpy() if isMDP else np.vstack([real_measurement, np.array(policy.measurement)]).T
        total_feed = comupute_cumulative_feed(feed)
        reward[b] = compute_reward(total_feed, states[-1,:], time_measurement[-1])
    return np.mean(reward)