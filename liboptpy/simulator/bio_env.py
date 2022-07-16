import numpy as np

from liboptpy.simulator.bioprocess_2stages import fermentation_purification_simulator

from liboptpy.simulator.bioprocess import bioprocess_simulator



class fermentation_env2():
    ''' fermentation simulation environment
    '''

    def __init__(self, obj, initial_state=[0.05, 0, 0, 30, 5, 0.7]):
        self.done = False
        self.initial_state = initial_state
        self.state = initial_state + [0]
        self.obj = obj
        self.glucose_consumed = 0
        self.glucose_added = 0  # self.initial_state[3]
        self.b = -0.13363 * 1000

    def reset(self):
        self.done = False
        self.state = self.initial_state + [0]
        self.glucose_consumed = 0
        self.glucose_added = 0  # self.initial_state[3]
        return self.state

    def compute_reward(self, cumulative_feed, state):
        titer = state[1]
        total_CA = state[1] * state[-1]
        oil_consumed = cumulative_feed * 917 + self.initial_state[3] * state[-1] - state[3] * state[-1]
        yield_CA = total_CA / oil_consumed
        pv = titer / self.obj.harvest_time
        man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
        total_reward = - man_cost * self.obj.harvest_time  # * state[-1] * state[1]
        return total_reward

    def step(self, action):
        next_state = self.obj.predict(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4],
                                      self.state[5], action[0], self.state[-1] * 4) + [self.state[-1] + 1]
        self.glucose_added += action[0] * self.obj.delta_t
        self.done = True if self.obj.harvest_time / self.obj.delta_t + 1 == next_state[-1] else False
        reward = self.b * action[0] + 1.28739 * (next_state[1] - self.state[1])
        # if self.done:
        #     reward = self.compute_reward(self.glucose_added - action[0], self.state[:-1])
        # else:
        #     reward = 0

        self.state = next_state
        return next_state, reward, self.done

class two_unit_operations_env():
    ''' fermentation simulation environment
    '''

    def __init__(self, obj, bn, time_measurement, real_measurement, m=None, b=None, c=None, ps=None, initial_state=[0.05, 0, 0, 30, 5, 0.7]):
        self.done = False
        self.initial_state = initial_state
        self.state = initial_state
        self.simulator = obj
        self.state_indices = [0, 1, 3, 4, 5]
        self.bn = bn
        self.ps = ps
        self.time_measurement = time_measurement
        self.real_measurement = real_measurement

        self.m = m
        self.b = b
        self.c = c

    def generate_trajectories(self, policy, r):
        trajectories = []
        total_rewards = []
        while len(trajectories) < r:
            success, trajectory, total_reward = self._generate_trajectory(policy)
            if not success:
                continue
            trajectories.append(trajectory)
            total_rewards.append(total_reward)
        data = np.vstack(trajectories)
        return data, total_rewards

    def _generate_trajectory(self, policy):
        init_states = self.initial_state + np.abs(
            np.random.normal(0, np.array(self.initial_state) / 10 + 0.01))


        simulator = fermentation_purification_simulator(self.bn, self.time_measurement, policy, self.real_measurement)
        result = simulator.g(self.time_measurement, init_states, self.ps)
        state = result[-1, :]
        result = result[:, self.state_indices]
        trajectory = []
        for t in range(len(self.time_measurement)):
            if t not in simulator.feed.keys() and (
                    t + 1 not in simulator.feed.keys() or t - 1 not in simulator.feed.keys()):
                return False, None, None
            if t not in simulator.feed.keys():
                simulator.feed[t] = (simulator.feed[t + 1] + simulator.feed[t - 1]) / 2
                trajectory.append((simulator.feed[t + 1] + simulator.feed[t - 1]) / 2)
            else:
                trajectory.append(simulator.feed[t])
            trajectory.append(result[t, 0])
            trajectory.append(result[t, 1])
            trajectory.append(result[t, 2])
            trajectory.append(result[t, 3])
            trajectory.append(result[t, 4])
            # trajectory = np.concatenate((trajectory, simulator.feed[t], result[t,:]))

        cumulative_feed = 0
        for t, time in enumerate(simulator.time_measurement[:-1]):
            cumulative_feed += simulator.feed[t] * (simulator.time_measurement[t + 1] - simulator.time_measurement[t])

        ammonium_sulphate, final_state, result2 = simulator.purification_processes(state)
        trajectory = np.concatenate([trajectory, result2])
        total_reward = self.compute_reward(cumulative_feed, ammonium_sulphate, final_state)
        return True, trajectory, total_reward

    def compute_reward(self, cumulative_feed, ammonium_sulphate, state):
        # titer = state[1]
        # total_CA = state[1] * state[-1]
        # oil_consumed = cumulative_feed * 917 - state[2] * state[-1]
        # yield_CA = total_CA / oil_consumed
        # pv = titer / self.time_measurement[-1]
        # man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
        # total_reward = - man_cost * self.time_measurement[-1] # * state[-1] * state[1]
        total_reward =  self.b[0][0] * cumulative_feed - self.b[-1][0] * sum(ammonium_sulphate) + self.c[self.bn.n_time - 1, 1] * state[0] - self.c[self.bn.n_time - 1, 2] * state[1]

        return total_reward



class fermentation_env():
    ''' fermentation simulation environment
    '''

    def __init__(self, obj, bn, time_measurement, real_measurement, m=None, b=None, c=None, ps=None, initial_state=[0.05, 0, 0, 30, 5, 0.7]):
        self.done = False
        self.initial_state = initial_state
        self.state = initial_state
        self.simulator = obj
        self.state_indices = [0, 1, 3, 4, 5]
        self.bn = bn
        self.ps = ps
        self.time_measurement = time_measurement
        self.real_measurement = real_measurement

        self.m = m
        self.b = b
        self.c = c

    def reset(self):
        self.done = False
        self.state = self.initial_state + [0]
        self.glucose_consumed = 0
        self.glucose_added = self.initial_state[3]
        return self.state

    def dissolve_oxygen(self, t):
        oxygen_data = [98.58750, 71.34750, 43.57285, 28.51669, 24.61062, 24.58139, 28.96252, 30.63478, 24.34490,
                       29.33842, 31.89646, 34.51994, 36.03211, 36.16354]

        index = 0
        time_w = -1
        for i, time in enumerate(self.simulator.measurement):
            if time - t > 0:
                index = i - 1
                time_w = time
                break
        return oxygen_data[index]

    def _step(self, state, action, t):
        next_state = self.simulator.one_step_predict(self.state[0], self.state[1], self.state[2], self.state[3],
                                                     self.state[4], self.state[5], action, self.dissolve_oxygen(t))
        glucose_added = action * self.simulator.dt
        return next_state, glucose_added

    # def generate_trajectory(self, policy):
    #     state = self.initial_state
    #     cumulative_feed = 0
    #     trajectory = np.array([])
    #     for t, time in enumerate(self.simulator.measurement[:-1]):
    #         partial_state = [state[i] for i in self.state_indices]
    #         partial_state = (partial_state - self.bn.mu[(self.bn.n_factor * (t + 1) - self.bn.num_state):(
    #                 self.bn.n_factor * (t + 1))]) / \
    #                 self.bn.sample_sd[(self.bn.n_factor * (t + 1) - self.bn.num_state):(self.bn.n_factor * (t + 1))]
    #         action = policy.next_action(partial_state, t)
    #         action = action if np.isscalar(action) else action[0] # single action
    #         trajectory = np.concatenate((trajectory, [action], partial_state))
    #         state, glucose_added = self.step(state, action, t)
    #         cumulative_feed += glucose_added
    #     action = 0
    #     trajectory = np.concatenate((trajectory, [action], [state[i] for i in self.state_indices]))
    #     titer = state[1]
    #     total_CA = state[1] * state[-1]
    #     oil_consumed = cumulative_feed * self.simulator.S_F + self.initial_state[2] * state[-1] - state[2] * state[-1]
    #     yield_CA = total_CA / oil_consumed
    #     pv = titer / self.simulator.measurement[-1]
    #     man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
    #     total_reward = - man_cost * self.simulator.measurement[-1] * state[-1] * state[1]
    #     return trajectory, total_reward

    def generate_trajectories(self, policy, r):
        trajectories = []
        total_rewards = []
        while len(trajectories) < r:
            success, trajectory, total_reward = self._generate_trajectory(policy)
            if not success:
                continue
            trajectories.append(trajectory)
            total_rewards.append(total_reward)
        data = np.vstack(trajectories)
        return data, total_rewards

    def _generate_trajectory(self, policy):
        init_states = self.initial_state + np.abs(
            np.random.normal(0, np.array(self.initial_state) / 10 + 0.01))

        # exploration vs exploitation
        # if np.random.uniform(0,1) < 0.2:
        #     theta = np.random.uniform(low=0., high=0.3, size=(self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        #     policy = Policy(theta, True)

        simulator = bioprocess_simulator(self.bn, self.time_measurement, policy, self.real_measurement)
        result = simulator.g(self.time_measurement, init_states, self.ps)
        result = result[:, self.state_indices]
        state = result[-1, :]
        trajectory = []
        for t in range(len(self.time_measurement)):
            if t not in simulator.feed.keys() and (
                    t + 1 not in simulator.feed.keys() or t - 1 not in simulator.feed.keys()):
                return False, None, None
            if t not in simulator.feed.keys():
                simulator.feed[t] = (simulator.feed[t + 1] + simulator.feed[t - 1]) / 2
                trajectory.append((simulator.feed[t + 1] + simulator.feed[t - 1]) / 2)
            else:
                trajectory.append(simulator.feed[t])
            trajectory.append(result[t, 0])
            trajectory.append(result[t, 1])
            trajectory.append(result[t, 2])
            trajectory.append(result[t, 3])
            trajectory.append(result[t, 4])
            # trajectory = np.concatenate((trajectory, simulator.feed[t], result[t,:]))

        cumulative_feed = 0
        for t, time in enumerate(simulator.time_measurement[:-1]):
            cumulative_feed += simulator.feed[t] * (simulator.time_measurement[t + 1] - simulator.time_measurement[t])

        total_reward = self.compute_reward(cumulative_feed, state)
        return True, np.array(trajectory), total_reward

    def compute_reward(self, cumulative_feed, state):
        titer = state[1]
        total_CA = state[1] * state[-1]
        oil_consumed = cumulative_feed * 917 - state[2] * state[-1]
        yield_CA = total_CA / oil_consumed
        pv = titer / self.time_measurement[-1]
        man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
        total_reward = - man_cost * self.time_measurement[-1] # * state[-1] * state[1]
        # total_reward = -0.13363 * 1000 * cumulative_feed + 1.28739 * state[1]
        return total_reward
