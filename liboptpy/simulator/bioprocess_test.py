from lmfit import Parameters, Parameter
from scipy.integrate import odeint
import tensorflow as tf
# from liboptpy.benchmark_algorithms.DDPG import get_actor
from liboptpy.lib.mechanism_model import mechanism
from liboptpy.policy import Policy, PolicyLab, PolicyDDPG
from liboptpy.simulator.BaseSimulator import BaseSimulator


# oxygen_data =[100.00000,77.84821,55.80968,41.98889,36.34808,30.70728,25.74450,22.81581,24.32410,25.82329,
#                       27.31338,28.80348,29.91986,28.42026,27.21857,26.91063,26.60268,26.29474,25.98679,30.07243,
#                       34.15808,33.26775,32.37743,31.48710,31.94246,36.43486,38.16142,37.12214,36.08286,35.04358,
#                       34.00429,35.23737,36.47045,37.70353,38.93661,40.16969]

# feed_profile = [0.000000e+00, 2.725157e-05, 3.797773e-04, 1.106910e-03, 2.100225e-03, 3.093540e-03, 4.834362e-03,
#                 8.817708e-03, 8.844302e-03, 8.692792e-03, 8.363176e-03, 8.033560e-03, 7.534593e-03, 5.850172e-03,
#                 4.679597e-03, 5.050561e-03, 5.421526e-03, 5.792490e-03, 6.163455e-03, 4.378520e-03, 2.593585e-03,
#                 2.716121e-03, 2.838657e-03, 2.961194e-03, 2.956281e-03, 2.569018e-03, 2.298258e-03, 2.144001e-03,
#                 1.989744e-03, 1.835487e-03, 1.681230e-03, 1.430548e-03, 1.179867e-03, 9.291858e-04, 6.785046e-04,
#                 4.278233e-04]

# feed_profile = [0.000000e+00, 3.131863e-05, 8.332656e-04, 3.641692e-03, 9.661720e-03, 7.794840e-03, 7.359495e-03,
#                 5.043735e-03, 5.347406e-03, 2.486022e-03, 2.385212e-03, 2.301092e-03, 1.724154e-03, 6.246971e-04]
# feed_profile = [0.0000000000, 0.0003131863, 0.0041663280, 0.0098325692, 0.0087833820, 0.0077948404, 0.0056611497,
#                 0.0038797959, 0.0041133889, 0.0019123248, 0.0015901411, 0.0015340615, 0.0011494357, 0.0004164647]




feed_profile = [0.0000000000,
                0.0003624999,
                0.0048485723,
                0.0119220315,
                0.0102440275,
                0.0086513113,
                0.0061447560,
                0.0039552809,
                0.0048855832,
                0.0020873176,
                0.0017531639,
                0.0015830354,
                0.0010269718,
                0.0004710181]
feed_profile = [0.0000000000, 0.0003131863, 0.0041663280, 0.0098325692, 0.0087833820, 0.0077948404, 0.0056611497, 0.0038797959, 0.0041133889, 0.0019123248, 0.0015901411, 0.0015340615, 0.0011494357, 0.0004164647]


# feed_profile = [0, 4.440000e-19, 0.002515011, 0.004502748, 0.005973753, 0.007297723, 0.006291863,  5.175258e-03, 0.0025989402, 0.001611073, 0.001167248, 0.001135921, 0.0010017584, 0.0001669597]
# feed_profile = [0.000000e+00,  0.000000e+00,  2.801345e-03,  6.500561e-03,  7.838870e-03,  7.379049e-03,  5.426248e-03, -4.236836e-05,
#  1.575191e-02,  1.355505e-03, -1.190099e-03,  1.957627e-03,  9.760737e-04,  1.626789e-04]
real_measurement = [0.0, 5.0, 10.0, 23.0, 28.0, 34.0, 47.5, 55.0, 72.0, 80.0, 95.0, 102.0, 120.0, 140.0]


class bioprocess_simulator(BaseSimulator):
    def __init__(self, bn, time_measurement, policy, real_measurement):
        self.bn = bn
        # self.feed = [0] * bn.n_time
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
        # print(t)
        t_index = t/ (140/(self.bn.n_time-1))
        print(t)
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
        # F_S, O = self.feed_rate(t, xs), self.dissolve_oxygen(t)
        # self.feed_every_second.append([t, F_S])
        # self.feed[int(t_index)] = F_S

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


if __name__ == '__main__':
    from liboptpy.gradients.bayesian_network import bayesian_network, bayesian_network_posterior
    from liboptpy.gradients.BN_gradient import BN_gradient
    import liboptpy.constr_solvers._proj_gd as cs
    import liboptpy.step_size as ss
    import numpy as np
    import pandas as pd
    from liboptpy.simulator.bio_env import fermentation_env, fermentation_env2
    from liboptpy.lib.compute_reward import compute_avg_reward

    params = Parameters()

    params.add('alpha_L', value=0.127275, min=0, max=0.5)
    params.add('c_max', value=130.901733, min=100, max=250)
    params.add('K_iN', value=0.12294103, min=0, max=5.064)
    params.add('K_iS', value=612.178130, min=300, max=700)
    params.add('K_iX', value=59.9737695, min=20, max=100)
    params.add('K_N', value=0.02000201, min=0, max=1)
    params.add('K_O', value=0.33085322, min=0, max=10)
    params.add('K_S', value=0.0430429, min=0, max=5)
    params.add('K_SL', value=0.02165744, min=0, max=1)
    params.add('m_s', value=0.02252332, min=0, max=2)
    params.add('r_L', value=0.47917813, min=0, max=1)
    params.add('V_evap', value=2.6 * 1e-03, min=0, max=10)
    params.add('Y_cs', value=0.682572, min=0, max=4)
    params.add('Y_ls', value=0.3574429, min=0, max=2)
    params.add('Y_xn', value=10.0, min=0, max=15)
    params.add('Y_xs', value=0.2385559, min=0, max=1)
    params.add('beta_LC_max', value=0.14255192, min=0, max=5)
    params.add('mu_max', value=0.3844627, min=0, max=2)
    params_copy = params.copy()


    beta_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/beta_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    v2_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/v2_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None,
                            dtype=np.float64)
    the_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/the_R_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    tau2_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/tau2_R_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    kap_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/kap_R_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    lam_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/lam_R_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    mu = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/mu_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)
    sd = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/sd_s5a1-R8-explore0.3-v1-modelrisk--ntime8.txt', header=None, dtype=np.float64)

    bn_post = bayesian_network_posterior(the_R.to_numpy(), tau2_R.to_numpy(), kap_R.to_numpy(), lam_R.to_numpy(),
                                         mu.to_numpy().flatten(), 5, 1, 8, beta_gibbs, v2_gibbs)
    p_beta, p_v2, mu = bn_post.posterior_sample(5000, useFixed=False)
    beta = np.apply_along_axis(np.mean, 2, p_beta)
    v2 = np.apply_along_axis(np.mean, 1, p_v2)
    bn = bayesian_network(mu, beta, v2, 1, 5, 8, True, mu, sd.to_numpy().flatten())


    ''' projected stochastic gradient accent with posterior sampling '''
    #for id in range(30):
    np.random.seed(117)
    time_interval = 140 / (bn.n_time - 1)
    time_measurement = [time_interval * i for i in range(bn.n_time)]
    flag_estimate_reward_fun = False
    m = [0] * (bn.n_time - 1) + [-15]  # $/g
    b = np.transpose(np.array([[-0.1 * 1000 * time_interval] * (bn.n_time - 1)]))  # $/g
    c = np.zeros(shape=(bn.n_time, bn.num_state))
    c[bn.n_time - 1, 1] = 1.28739  # product revenue
    c[bn.n_time - 1, 0] = 0  # purification cost
    c[bn.n_time - 1, 2] = 0  # purification cost


    # c[13, 2] = -0.2 # purification cost
    gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
     # 1972 # 1983 (0.2)
    theta = np.random.uniform(low=0., high=0.3, size=(bn.n_time - 1, bn.num_state, bn.num_action))

    grad = lambda x: - gradient.grad_f(x)
    f = lambda x: - gradient.func(x)
    def projection(y):
        theta = y.reshape((bn.n_time - 1, bn.num_state, bn.num_action))
        theta[:, 0, :] = np.clip(theta[:, 0, :], 0, 0.3)
        theta[:, 1, :] = np.clip(theta[:, 0, :], 0, 0.3)
        theta[0, 2, :] = np.clip(theta[0, 2, :], -0.1, 0.1)
        theta[:, 2, :] = np.clip(theta[:, 2, :], -0.2, 0.2)
        theta[:, 3, :] = np.clip(theta[:, 3, :], -0.1, 0.02)
        theta[:, 4, :] = np.clip(theta[:, 4, :], -0.7, 0.5)
        return theta.flatten()

    process_model = mechanism()
    bio_env = fermentation_env(process_model, bn, time_measurement, real_measurement, params)
    methods = {"PSGD": cs.ProjectedSGD(f,
                                     grad,
                                     projection,
                                     # ss.InvSqrootIterStepSize(),
                                     ss.Backtracking(rule_type="Armijo", rho=0.3, beta=0.1, init_alpha=1.),
                                     bn_post,
                                     sd.to_numpy().flatten(),
                                     bio_env,
                                     batch_size=8,
                                     B=25)
               }
    max_iter = 100
    tol = 1e-25
    for m_name in methods:
        print("\t", m_name)
        x = methods[m_name].solve(x0=theta.flatten(), max_iter=max_iter, tol=tol, disp=2)
    # gradient.states
    # [f(x) for x in methods[m_name].get_convergence()]

    theta = x.reshape((bn.n_time - 1, bn.num_state, bn.num_action))

    policy = Policy(theta, True)


    # ### Bayesian network prediction
    # transitions = []
    # states = []
    # current_transit = bn.initial_state_base[:-1] * bn.initial_state_base[
    #     -1]  # * bn.sample_sd[bn.num_action: bn.n_factor] #+  - bn.mu[bn.num_action: bn.n_factor]  # s- mu_s
    # states.append(current_transit)
    # for t in range(bn.n_time - 1):
    #     transitions.append(bn.beta_state[t].T + bn.beta_action[t].T @ theta[t].T)
    #     current_transit = transitions[-1] @ current_transit
    #     states.append(current_transit)
    # reward = 0
    # for t in range(bn.n_time):
    #     if t != bn.n_time - 1:
    #         a_t = bn.mu_a[t] + np.matmul(theta[t].T, states[t])
    #         reward += m[t] + np.dot(c[t], bn.rescale_state(states[t], t)) + np.dot(b[t], bn.rescale_action(a_t, t))
    #     else:
    #         reward += m[t] + np.dot(c[t], bn.rescale_state(states[t], t))
    # print(reward)
    # bn.rescale_state(gradient.states[34], 34)

    '''bn
    simulate new policy and existing policy with true model (ODEs)
    '''
    ## use BN_MDP policy
    B = 50
    x0 = bn.initial_state_full
         #+ np.abs(
         #   np.random.normal(0, np.array(bn.initial_state_full) / 10 + 0.01))
    # time_measurement = [4 * i for i in range(bn.n_time)]
    # time_measurement[-1] = 135.9
    simulator = bioprocess_simulator(bn, time_measurement, policy, real_measurement)
    model = simulator.g(time_measurement, x0, params)
    print(model)

    # compute_avg_reward(bn, params, policy, time_measurement, real_measurement, B, isMDP=True)


    ## existing policy
    # policy_lab = PolicyLab(real_measurement, feed_profile, False)
    # simulator_lab = bioprocess_simulator(bn, real_measurement, policy_lab, real_measurement)
    # model_lab = simulator_lab.g(time_measurement, x0, params)
    # model_lab
    # pd.DataFrame(model_lab).to_csv('lab1-v1.csv', index=False)
    # compute_avg_reward(bn, params, policy_lab, time_measurement, real_measurement, 100, isMDP=False)

    ## existing policy
    # time_measurement = [4 * i for i in range(bn.n_time)]
    # upper_bound = 0.02
    # lower_bound = 0.0



    # simulator.feed
    # print(model[34][1] * model[34][-1])
    # plt.plot(simulator.feed)
    # plt.show()
    # plt.plot(model[:,3])
    # plt.show()
    # # plot with various axes scales
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # variables = ['Cell Mass', 'CA', 'Lipid', 'Oil', 'Nitrogen', 'V']
    # i=0
    # plt.plot(time_measurement, model[:, i], linestyle='-.', color=colors[i], linewidth=2)
    # plt.xlabel('hours')
    # plt.ylabel(variables[i])
    # plt.savefig('RL-{}.png'.format(variables[i]))
    # plt.show()
    #
    # plt.plot(time_measurement, model_lab, linestyle='-')
    # plt.legend(['Cell Mass', 'CA', 'Lipid', 'Oil', 'Nitrogen', 'V'], loc='upper left')
    #
    # plt.show()
    # pd.DataFrame(model).to_csv('BN-MDP-R400-v1.csv', index=False)
    # pd.DataFrame(model_lab).to_csv('lab1.csv', index=False)
    #
    # for i, time in enumerate(time_measurement):
    #     if time - t > 0:
    #         index = i - 1
    #         time_w = time
    #         break


    # np.sum(list(simulator.feed.values())) * 4
    #
    # feed_BN = pd.DataFrame({"time": np.array(list(simulator.feed.keys())) * 4, "feed_rate": list(simulator.feed.values())}, dtype="float64")
    # feed_BN = pd.DataFrame(simulator.feed_every_second, columns=['time', 'feed_rate'], dtype="float64").drop_duplicates()
    #
    # feed_BN = feed_BN.sort_values(by='time')
    # feed_BN.round({'time': 0}).plot.bar(x='time', y='feed_rate')
    # plt.show()
    # feed_human1 = pd.DataFrame(simulator_lab.feed_every_second, columns=['time', 'feed_rate'], dtype="float64").drop_duplicates()
    # feed_human1 = feed_human1.sort_values(by='time')
    # feed_human1.round({'time': 0}).plot.bar(x='time', y='feed_rate')
    # plt.show()
    #
    # total_feed_MDP = 0
    # for i in range(feed_BN.shape[0]-1):
    #     total_feed_MDP += (feed_BN.iloc[i+1][0] - feed_BN.iloc[i][0]) * (feed_BN.iloc[i][1])
