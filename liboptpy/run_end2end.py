from lmfit import Parameters, Parameter
from liboptpy.lib.mechanism_model import mechanism
from liboptpy.policy import Policy, PolicyLab, PolicyDDPG
from liboptpy.gradients.bayesian_network import bayesian_network, bayesian_network_posterior
from liboptpy.gradients.BN_gradient import BN_gradient
import liboptpy.constr_solvers._proj_gd as cs
import liboptpy.step_size as ss
import numpy as np
import pandas as pd
from liboptpy.simulator.bio_env import fermentation_env, fermentation_env2, two_unit_operations_env
from liboptpy.lib.compute_reward import compute_avg_reward
from liboptpy.simulator.bioprocess_2stages import fermentation_purification_simulator


feed_profile = [0.000000e+00, 3.131863e-05, 8.332656e-04, 3.641692e-03, 9.661720e-03, 7.794840e-03, 7.359495e-03,
                5.043735e-03, 5.347406e-03, 2.486022e-03, 2.385212e-03, 2.301092e-03, 1.724154e-03, 6.246971e-04,np.log(40), np.log(60)]
# feed_profile = [0.0000000000, 0.0003131863, 0.0041663280, 0.0098325692, 0.0087833820, 0.0077948404, 0.0056611497,
#                 0.0038797959, 0.0041133889, 0.0019123248, 0.0015901411, 0.0015340615, 0.0011494357, 0.0004164647]

#
# feed_profile = [0.0000000000,
#                 0.0003624999,
#                 0.0048485723,
#                 0.0119220315,
#                 0.0102440275,
#                 0.0086513113,
#                 0.0061447560,
#                 0.0039552809,
#                 0.0048855832,
#                 0.0020873176,
#                 0.0017531639,
#                 0.0015830354,
#                 0.0010269718,
#                 0.0004710181,
#                 np.log(40), np.log(60)]
# feed_profile = [0.0000000000, 0.0003131863, 0.0041663280, 0.0098325692, 0.0087833820, 0.0077948404, 0.0056611497,
#                 0.0038797959, 0.0041133889, 0.0019123248, 0.0015901411, 0.0015340615, 0.0011494357, 0.0004164647,
#                 np.log(40), np.log(60)]

# feed_profile = [0, 4.440000e-19, 0.002515011, 0.004502748, 0.005973753, 0.007297723, 0.006291863,  5.175258e-03, 0.0025989402, 0.001611073, 0.001167248, 0.001135921, 0.0010017584, 0.0001669597]
# feed_profile = [0.000000e+00,  0.000000e+00,  2.801345e-03,  6.500561e-03,  7.838870e-03,  7.379049e-03,  5.426248e-03, -4.236836e-05,
#  1.575191e-02,  1.355505e-03, -1.190099e-03,  1.957627e-03,  9.760737e-04,  1.626789e-04]
real_measurement = [0.0, 5.0, 10.0, 23.0, 28.0, 34.0, 47.5, 55.0, 72.0, 80.0, 95.0, 102.0, 120.0, 140.0]



if __name__ == '__main__':



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

    beta_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/beta_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    v2_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/v2_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None,
                            dtype=np.float64)
    the_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/the_R_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    tau2_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/tau2_R_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    kap_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/kap_R_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    lam_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/lam_R_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    mu = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/mu_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    sd = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/sd_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    bn_post = bayesian_network_posterior(the_R.to_numpy(), tau2_R.to_numpy(), kap_R.to_numpy(), lam_R.to_numpy(),
                                         mu.to_numpy().flatten(), 5, 1, 39, beta_gibbs, v2_gibbs)
    p_beta, p_v2, mu = bn_post.posterior_sample(5000, useFixed=False)
    beta = np.apply_along_axis(np.mean, 2, p_beta)
    v2 = np.apply_along_axis(np.mean, 1, p_v2)
    bn = bayesian_network(mu, beta, v2, 1, 5, 39, True, mu, sd.to_numpy().flatten())

    ''' set up with the constraints '''
    def projection(y):
        theta = y.reshape((bn.n_time - 1, bn.num_state, bn.num_action))
        theta[:bn.n_time - 1 - 2, 0, :] = np.clip(theta[:bn.n_time - 1 - 2, 0, :], 0, 0.6)
        theta[:bn.n_time - 1 - 2, 1, :] = np.clip(theta[:bn.n_time - 1 - 2, 0, :], 0, 0.6)
        theta[0, 2, :] = np.clip(theta[0, 2, :], -0.2, 0.2)
        theta[:bn.n_time - 1 - 2, 2, :] = np.clip(theta[:bn.n_time - 1 - 2, 2, :], -0.4, 0.4)
        theta[:bn.n_time - 1 - 2, 3, :] = np.clip(theta[:bn.n_time - 1 - 2, 3, :], -0.2, 0.04)
        theta[:bn.n_time - 1 - 2, 4, :] = np.clip(theta[:bn.n_time - 1 - 2, 4, :], -1.4, 1)

        theta[bn.n_time - 1 - 2, 0, :] = np.clip(theta[bn.n_time - 1 - 2, 0, :], -0.1, 0.02)
        theta[bn.n_time - 1 - 2, 1, :] = np.clip(theta[bn.n_time - 1 - 2, 0, :], -0.02, 0.1)

        theta[bn.n_time - 1 - 1, 0, :] = np.clip(theta[bn.n_time - 1 - 1, 0, :], -0.0, 0.1)
        theta[bn.n_time - 1 - 1, 1, :] = np.clip(theta[bn.n_time - 1 - 1, 0, :], -0.1, 0.0)

        theta[bn.n_time - 1 - 2:, 2, :] = 0
        theta[bn.n_time - 1 - 2:, 3, :] = 0
        theta[bn.n_time - 1 - 2:, 4, :] = 0
        return theta.flatten()


    ''' projected gradient accent with posterior sampling '''

    cumulative_rewards = []
    cumulative_profits =[]
    final_purity = []
    final_titer = []
    total_feeding = []
    feed_profiles = []
    conv_trajectories = []
    patient = 15
    window = 15
    max_iter = 100
    algorithm = 'PGA'
    fixedBN = False
    for id in range(30):
        feed_profile_ddpg = []
        np.random.seed(1 + id)

        time_interval = 140 / (bn.n_time - 3 - 1)
        time_measurement = [time_interval * i for i in range(bn.n_time -3)]
        flag_estimate_reward_fun = False
        m = [0] * (bn.n_time - 1) + [-15]  # $/g
        b = np.transpose(np.array([[-0.13363 * 1000 * time_interval] * (bn.n_time - 3) + [-0.05] * 2]))  # $/g
        c = np.zeros(shape=(bn.n_time, bn.num_state))
        c[bn.n_time - 1, 0] = 1.28739  # product revenue
        c[bn.n_time - 1, 1] = - 1  # purification cost


        gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
        theta = np.random.uniform(low=0., high=0.3, size=(bn.n_time - 1, bn.num_state, bn.num_action))
        grad = lambda x: - gradient.grad_f(x)
        f = lambda x: - gradient.func(x)



        process_model = mechanism()
        bio_env = two_unit_operations_env(process_model, bn, time_measurement, real_measurement, m, b, c, params)
        if not fixedBN:
            methods = {"PGA": cs.ProjectedGDWithPosterior(f,
                                                          grad,
                                                          projection,
                                                          ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1,
                                                                          init_alpha=0.001),
                                                          bn_post,
                                                          bio_env,
                                                          sd.to_numpy().flatten(),
                                                          256,
                                                          patient,
                                                          window)
                       }
        else:
            methods = {"PGA": cs.ProjectedBNPG(f,
                                               grad,
                                               projection,
                                               ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1,
                                                               init_alpha=0.001),
                                               bn_post,
                                               bio_env,
                                               sd.to_numpy().flatten(),
                                               256,
                                               patient,
                                               window)
                       }

        tol = -1000  # 1e-25
        for m_name in methods:
            print("\t", m_name)
            x = methods[m_name].solve(x0=theta.flatten(), max_iter=max_iter, tol=tol, disp=2)
        # gradient.states
        # [f(x) for x in methods[m_name].get_convergence()]

        theta = x.reshape((bn.n_time - 1, bn.num_state, bn.num_action))
        policy = Policy(theta, True)
        '''bn
        simulate new policy and existing policy with true model (ODEs)
        '''
        B = 50
        x0 = bn.initial_state_full

        simulator = fermentation_purification_simulator(bn, time_measurement, policy, real_measurement)
        state = simulator.g(time_measurement, x0, params)
        state = state[-1, :]
        feed_profile_ddpg = list(simulator.feed.values())
        total_feed_MDP = np.sum(feed_profile_ddpg)
        ammonium_sulphate, downstream_state, result2 = simulator.purification_processes(state)

        conv_trajectories.append(methods[m_name].convergence_reward)

        cur_titer = np.exp(downstream_state[0])
        cur_purity = np.exp(downstream_state[0]) / (np.exp(downstream_state[1]) + np.exp(downstream_state[0]))
        final_titer.append(cur_titer)
        final_purity.append(cur_purity)
        feed_profiles.append(list(simulator.feed.values()))
        total_feeding.append(total_feed_MDP * (140 / (bn.n_time - 3 - 1)))
        # final_reward = -0.13363 * 1000 * total_feed_MDP + 1.28739 * state[-1, 1]
        final_reward = -15 - 0.13363 * 1000 * total_feed_MDP - b[-1][0] * sum(ammonium_sulphate) \
                       + c[bn.n_time - 1, 0] * downstream_state[0] + c[bn.n_time - 1, 1] * downstream_state[1]
        final_profit = -15 - 0.13363 * 1000 * total_feed_MDP - b[-1][0] * sum(np.exp(ammonium_sulphate)) \
                       + c[bn.n_time - 1, 0] * np.exp(downstream_state[0]) + c[bn.n_time - 1, 1] * np.exp(
            downstream_state[1])

        cumulative_rewards.append(final_reward)
        cumulative_profits.append(final_profit)
        print('--------------------------------------------------')
        print(id)
        # print(policy.theta)
        print("mean final titer = {}".format(np.exp(downstream_state[0])))
        print("mean final purity = {}".format(np.exp(downstream_state[0])/(np.exp(downstream_state[1]) + np.exp(downstream_state[0]))))
        print("mean reward = {}".format(final_reward))
        print("mean profit = {}".format(final_profit))

    print('---------------------Summary-------------------------')
    print(np.mean(final_titer), np.std(final_titer))
    print(np.mean(final_purity), np.std(final_purity))
    print(np.mean(cumulative_rewards), np.std(cumulative_rewards))
    print(np.mean(cumulative_profits), np.std(cumulative_profits))


