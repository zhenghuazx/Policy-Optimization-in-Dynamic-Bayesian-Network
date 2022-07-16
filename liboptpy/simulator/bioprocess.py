
from scipy.integrate import odeint
from liboptpy.simulator.BaseSimulator import BaseSimulator


class bioprocess_simulator(BaseSimulator):
    def __init__(self, bn, time_measurement, policy, real_measurement):
        self.bn = bn
        # self.feed = [0] *  bn.n_time
        self.oxygen = [0] * 14 # bn.n_time
        self.current_time = -1
        self.cur_action = None
        self.policy = policy
        self.feed = {}
        self.feed_every_second = []
        self.time_measurement = time_measurement
        self.real_measurement = real_measurement

    def feed_rate(self, t, xs):
        state_index = [0, 1, 3, 4] if self.bn.num_state == 4 else range(5)
        if self.bn.num_state == 5 and self.bn.num_action == 1:
            state_index = [0, 1, 3, 4, 5]
        index = 0
        time_w = -1
        for i, time in enumerate(self.time_measurement):
            if time - t > 0:
                index = i - 1
                time_w = time
                break

        ### model-free RL
        state = xs
        if self.policy.policy_algo == 'DDPG':
            a_t = self.policy.next_action(state, index)
            print(a_t[0], index)
            return a_t[0]

        ### human policy

        if not self.policy.scale_needed:
            a_t = self.policy.next_action([], t)
            # self.feed[self.current_time] = a_t
            return a_t

        ### BN-MDP
        state = (state[state_index] - self.bn.mu[(self.bn.n_factor * (index + 1) - self.bn.num_state):(
                    self.bn.n_factor * (index + 1))]) / \
                self.bn.sample_sd[(self.bn.n_factor * (index + 1) - self.bn.num_state):(self.bn.n_factor * (index + 1))]
        # print(index)

        self.current_time = index
        if index in self.feed.keys():
            return self.feed[self.current_time]

        if self.policy.scale_needed:
            a_t = self.policy.next_action(state, index)
            action = self.bn.rescale_action(a_t, index)
            if action[0] < 0:
                action[0] = 0
            if action[0] > 0.02:
                action[0] = 0.02
            if len(action) > 1 and action[1] < 0:
                action[1] = 0
            if len(action) > 1 and action[1] > 100:
                action[1] = 100
            # self.feed[self.current_time] = action[0]
            return action[0]


    def dissolve_oxygen(self, t):
        oxygen_data = [98.58750, 71.34750, 43.57285, 28.51669, 24.61062, 24.58139, 28.96252, 30.63478, 24.34490,
                       29.33842, 31.89646, 34.51994, 36.03211, 36.16354]

        index = 0
        time_w = -1
        for i, time in enumerate(self.real_measurement):
            if time - t > 0:
                index = i - 1
                time_w = time
                break
        # index = 1
        self.oxygen[index] = oxygen_data[index]
        return oxygen_data[index]

    def f(self, xs, t, ps):
        """bioreactor model."""
        alpha_L = ps['alpha_L'].value
        c_max = ps['c_max'].value
        K_iN = ps['K_iN'].value
        K_iS = ps['K_iS'].value
        K_iX = ps['K_iX'].value
        K_N = ps['K_N'].value
        K_O = ps['K_O'].value
        K_S = ps['K_S'].value
        K_SL = ps['K_SL'].value
        m_s = ps['m_s'].value
        r_L = ps['r_L'].value
        #    V_evap = ps['V_evap'].value
        V_evap = ps['V_evap'].value
        Y_cs = ps['Y_cs'].value
        Y_ls = ps['Y_ls'].value
        Y_xn = ps['Y_xn'].value
        Y_xs = ps['Y_xs'].value

        beta_LC_max = ps['beta_LC_max'].value
        mu_max = ps['mu_max'].value
        # oil concentration in oil feed
        S_F = 917

        X_f, C, L, S, N, V = xs
        t_index = t/ (140/(self.bn.n_time-1))
        if int(t_index) in self.feed.keys():
            F_S, O = self.feed[int(t_index)], self.dissolve_oxygen(t)
        elif self.bn.num_action == 1:
            F_S, O = self.feed_rate(t, xs), self.dissolve_oxygen(t)
            self.feed_every_second.append([t, F_S])
            self.feed[int(t_index)] = F_S
        elif self.bn.num_action == 2:
            F_S, O = self.feed_rate(t, xs)
            self.feed_every_second.append([t, F_S])
            self.feed[int(t_index)] = F_S

        beta_LC = K_iN / (K_iN + N) * S / (S + K_S) * K_iS / (K_iS + S) * O / (K_O + O) * K_iX / (K_iX + X_f) * (
                    1 - C / c_max) * beta_LC_max
        q_C = 2 * (1 - r_L) * beta_LC
        beta_L = r_L * beta_LC - K_SL * L / (L + X_f) * O / (O + K_O)
        mu = mu_max * S / (S + K_S) * K_iS / (K_iS + S) * N / (K_N + N) * O / (K_O + O) / (1 + X_f / K_iX)
        q_S = 1 / Y_xs * mu + O / (O + K_O) * S / (K_S + S) * m_s + 1 / Y_cs * q_C + 1 / Y_ls * beta_L
        F_B = V / 1000 * (7.14 / Y_xn * mu * X_f + 1.59 * q_C * X_f)
        D = (F_B + F_S) / V
        # print("{0}:{1}:{2}:{3}".format(t,mu,q_C, F_S))
        dX_f = mu * X_f - (D - V_evap / V) * X_f
        dC = q_C * X_f - (D - V_evap / V) * C
        dL = (alpha_L * mu + beta_L) * X_f - (D - V_evap / V) * L
        dS = - (q_S * X_f - F_S / V * S_F + (D - V_evap / V) * S)
        dN = - (1 / Y_xn * mu * X_f + (D - V_evap / V) * N)
        dV = F_B + F_S - V_evap
        return [dX_f, dC, dL, dS, dN, dV]

    def g(self, t, x0, ps):
        """
        Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
        """
        x = odeint(self.f, x0, t, args=(ps,), rtol=1.49012e-106, mxstep=200000)
        return x


