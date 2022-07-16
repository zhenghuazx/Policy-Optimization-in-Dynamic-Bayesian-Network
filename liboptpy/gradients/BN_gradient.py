import numpy as np
import pandas as pd

from liboptpy.gradients.gradient import base_gradient
from liboptpy.gradients.bayesian_network import bayesian_network, bayesian_network_posterior
from liboptpy.policy import Policy


class BN_gradient(base_gradient):
    def __init__(self, m, b, c, bn, time_measurement=None, flag_estimate_reward_fun=False):
        self.m = m
        self.b = b
        self.c = c
        self.bn = bn
        self.cumulative_feed = 0
        self.flag_estimate_reward_fun = flag_estimate_reward_fun
        self.time_measurement = time_measurement

    def reset_cumulative_feed(self):
        self.cumulative_feed = 0

    def compute_normalized_state_one_step(self, theta, cur_state, t):
        next_state = (self.bn.beta_state[t].T + self.bn.beta_action[t].T @ theta[t].T) @ cur_state.T
        for i in range(next_state.shape[1]):
            next_state[:, i] = self.bn.rescale_state(next_state[:, i], t + 1, "standard")
        return next_state


    def compute_normalized_state(self, theta):
        transitions = []
        states = []
        current_transit = self.bn.initial_state  # - self.bn.mu[self.bn.num_action: self.bn.n_factor] # s- mu_s
        states.append(current_transit)
        self.states = states
        for t in range(self.bn.n_time - 1):
            transitions.append(self.bn.beta_state[t].T + self.bn.beta_action[t].T @ theta[t].T)
            current_transit = transitions[-1] @ current_transit
            states.append(current_transit)
        return states

    def func(self, theta):
        if len(theta.shape) == 1:
            theta = theta.reshape((self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        self.bn.initial_state_generator()
        states = self.compute_normalized_state(theta)
        reward = 0
        for t in range(self.bn.n_time):
            if t != self.bn.n_time - 1:
                a_t = np.matmul(theta[t].T, states[t])  # +self.bn.mu_a[t]
                # reward += self.m[t] + np.dot(self.c[t], self.bn.rescale_state(states[t], t)) + np.dot(self.b[t],
                #                                                                                       self.bn.rescale_action(
                #                                                                                           a_t, t))
                reward += np.dot(self.c[t], self.bn.rescale_state(states[t], t)) + np.dot(self.b[t],
                                                                                                      self.bn.rescale_action(
                                                                                                          a_t, t))
            else:
                reward += self.m[t] + np.dot(self.c[t], self.bn.rescale_state(states[t], t))
        return reward

    def grad_sgd(self, theta, data, total_rewards):
        '''
        :param theta: n by m
        :param data: r by (num_factor * n_time)
        :param total_rewards: (r by 1)
        :return:
        '''
        # states = self.compute_normalized_state(theta)
        # compute the state difference between true state and estimated state

        # if self.flag_estimate_reward_fun:
        #     self.reward_function(states, theta)
        theta = theta if len(theta.shape) == 3 else theta.reshape(
            (self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        grad = np.zeros(shape=(self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        for t in range(self.bn.n_time - 1):
            Vt = self.bn.v2[(self.bn.n_factor * (t+1) - self.bn.num_state):(self.bn.n_factor * (t+1))]
            term1 = self.bn.beta_action[t] @ np.diag(Vt)
            cur_state = data[:,(self.bn.n_factor * (t+1) - self.bn.num_state):(self.bn.n_factor * (t+1))]# data[:, (self.bn.n_factor * t):(self.bn.n_factor * t + self.bn.num_state)] # r by n
            next_state = data[:,(self.bn.n_factor * (t+2) - self.bn.num_state):(self.bn.n_factor * (t+2))]
            estimated_next_state = self.compute_normalized_state_one_step(theta, cur_state, t).T # r by n
            temporal_diff = next_state - estimated_next_state
            mean_difference = cur_state # cur_state - self.bn.sample_mean[(self.bn.n_factor * (t+1) - self.bn.num_state):(self.bn.n_factor * (t+1))]
            part = (mean_difference.T * total_rewards).T # [mean_difference[i] * total_rewards[i] for i in range(total_rewards.shape[0])] # mean_difference * total_rewards
            part1 = term1 @ temporal_diff.T @ part
            grad[t] = part1.T

        reward_grad = np.zeros(theta.flatten().shape)
        for r in range(data.shape[0]):
            reward_grad = reward_grad + self.grad_f_sgd(theta, data[r, :], total_rewards[r])

        g = grad.flatten() + reward_grad / data.shape[0]

        return g


    def grad_f_sgd(self, theta, data, total_rewards):
        derivatives = self.backpropagate_sgd(theta, data, total_rewards)
        grad = np.zeros(shape=(self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        for i in range(self.bn.n_time - 1):
            for j in range(i):
                grad[j] += derivatives[i][j]

        return grad.flatten()

    def grad_f(self, theta):
        derivatives = self.backpropagate(theta)
        grad = np.zeros(shape=(self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        for i in range(self.bn.n_time - 1):
            for j in range(i):
                grad[j] += derivatives[i][j]

        return grad.flatten()

    # def reward_function(self, states, theta):
    #     S_F = 917
    #     # manufacturing cost = 2/productivity+1/yield
    #     cumulative_feed = 0
    #     for t in range(self.bn.n_time):
    #         if t != self.bn.n_time - 1:
    #             a_t = np.matmul(theta[t].T, states[t])  # +self.bn.mu_a[t]
    #             cumulative_feed += a_t * (self.time_measurement[t + 1] - self.time_measurement[t + 1])
    #     final_state = self.bn.rescale_state(states[self.bn.n_time - 1], self.bn.n_time - 1)
    #     titer = final_state[1]
    #     total_CA = final_state[1] * final_state[-1]
    #     oil_consumed = cumulative_feed * S_F - final_state[2] * final_state[-1]
    #     yield_CA = total_CA / oil_consumed
    #     pv = titer / self.time_measurement[-1]
    #     man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
    #     self.c = np.zeros(shape=(self.bn.n_time, self.bn.num_state))
    #     self.c[self.bn.n_time - 1, 1] = - man_cost * self.time_measurement[-1] * final_state[-1]  # $/(g/L)
    #     self.m = [0] * self.bn.n_time
    #     self.b = np.zeros((self.bn.n_time - 1, 1))  # np.transpose(np.array([[0] * self.bn.n_time - 1]))

    def backpropagate_sgd(self, theta_flattented, data, total_rewards):
        theta = theta_flattented if len(theta_flattented.shape) == 3 else theta_flattented.reshape(
            (self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        self.bn.initial_state_generator()

        transitions = []
        states = []
        for t in range(self.bn.n_time):
            if t < self.bn.n_time-1:
                transitions.append(self.bn.beta_state[t].T + self.bn.beta_action[t].T @ theta[t].T)
            state = data[(self.bn.n_factor * (t + 1) - self.bn.num_state): (self.bn.n_factor * (t + 1))]
            states.append(state)

        # self.transitions = transitions
        delta_t = []
        for t in range(self.bn.n_time):
            deltas = []
            mul_deltas = []
            for k in reversed(range(t)):
                if t == self.bn.n_time - 1 and k == t:
                    continue
                mul_delta_tran = np.identity(self.bn.num_state) if len(deltas) == 0 else transitions[k + 1].T @ \
                                                                                         mul_deltas[-1]
                delta_tran = mul_delta_tran @ self.bn.beta_action[k].T
                mul_deltas.append(mul_delta_tran)
                deltas.append(delta_tran)

            delta_t.append(list(reversed(deltas)))

        # self.delta_t = delta_t
        # print(transitions[1].shape)
        if self.flag_estimate_reward_fun:
            self.reward_function(states, theta)
        derivatives = []
        for t in range(self.bn.n_time):
            unit_cost = self.c[t] + theta[t] @ self.b[t] if t != self.bn.n_time - 1 else self.c[t]
            p_derivatives = []
            for k in reversed(range(t)):
                if t == self.bn.n_time - 1 and k == t:
                    continue
                # print("{},{}".format(t,k))
                p_der = np.outer(states[k], self.b[k]) if k == t else np.outer(
                    unit_cost, states[k]) @ delta_t[t][k]
                p_derivatives.append(p_der)
            derivatives.append(list(reversed(p_derivatives)))
        # self.derivatives = derivatives
        return derivatives

    def backpropagate(self, theta_flattented, data = None, total_rewards = None):
        theta = theta_flattented if len(theta_flattented.shape) == 3 else theta_flattented.reshape(
            (self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        self.bn.initial_state_generator()

        transitions = []
        states = []
        current_transit = self.bn.initial_state  # - self.bn.mu[self.bn.num_action: self.bn.n_factor] # s- mu_s
        states.append(current_transit)
        self.states = states
        for t in range(self.bn.n_time - 1):
            transitions.append(self.bn.beta_state[t].T + self.bn.beta_action[t].T @ theta[t].T)
            current_transit = transitions[-1] @ current_transit
            states.append(current_transit)

        # self.transitions = transitions
        delta_t = []
        for t in range(self.bn.n_time):
            deltas = []
            mul_deltas = []
            for k in reversed(range(t)):
                if t == self.bn.n_time - 1 and k == t:
                    continue
                mul_delta_tran = np.identity(self.bn.num_state) if len(deltas) == 0 else transitions[k + 1].T @ \
                                                                                         mul_deltas[-1]
                delta_tran = mul_delta_tran @ self.bn.beta_action[k].T
                mul_deltas.append(mul_delta_tran)
                deltas.append(delta_tran)

            delta_t.append(list(reversed(deltas)))

        # self.delta_t = delta_t
        # print(transitions[1].shape)
        # if self.flag_estimate_reward_fun:
        #     self.reward_function(states, theta)
        derivatives = []
        for t in range(self.bn.n_time):
            unit_cost = self.c[t] + theta[t] @ self.b[t] if t != self.bn.n_time - 1 else self.c[t]
            p_derivatives = []
            for k in reversed(range(t)):
                if t == self.bn.n_time - 1 and k == t:
                    continue
                # print("{},{}".format(t,k))
                # p_der = np.outer(self.bn.rescale_state(states[k], k, "sd"), self.b[k]) if k == t else np.outer(
                #     unit_cost, self.bn.rescale_state(states[k], k, "sd")) @ delta_t[t][k]
                p_der = np.outer(states[k], self.b[k]) if k == t else np.outer(
                      unit_cost, states[k]) @ delta_t[t][k]
                p_derivatives.append(p_der)
            derivatives.append(list(reversed(p_derivatives)))
        # self.derivatives = derivatives
        return derivatives


if __name__ == '__main__':
    import liboptpy.constr_solvers as cs
    import liboptpy.step_size as ss
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # build bayesian network
    beta_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/beta_s5a1.txt', header=None,
                             dtype=np.float64)
    v2_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/v2_s5a1.txt', header=None,
                           dtype=np.float64)
    the_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/the_R_s5a1.txt', header=None, dtype=np.float64)
    tau2_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/tau2_R_s5a1.txt', header=None,
                         dtype=np.float64)
    kap_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/kap_R_s5a1.txt', header=None, dtype=np.float64)
    lam_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/lam_R_s5a1.txt', header=None, dtype=np.float64)
    mu = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/mu_s5a1.txt', header=None, dtype=np.float64)
    sd = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/sd_s5a1.txt', header=None, dtype=np.float64)

    bn_post = bayesian_network_posterior(the_R.to_numpy(), tau2_R.to_numpy(), kap_R.to_numpy(), lam_R.to_numpy(),
                                         mu.to_numpy().flatten(), 5, 1, 36, beta_gibbs, v2_gibbs)
    p_beta, p_v2, mu = bn_post.posterior_sample(100000, useFixed=False)
    beta = np.apply_along_axis(np.mean, 2, p_beta)
    v2 = np.apply_along_axis(np.mean, 1, p_v2)
    bn = bayesian_network(mu, beta, v2, 1, 5, 36, True, mu, sd.to_numpy().flatten())

    # m = [-2]* 13 + [0]  # $/g
    # b = np.transpose(np.array([[-0.5/1000] * 13, [0]*13])) # $/g
    # c = np.zeros(shape=(14,5))
    # c[13,1] = 1 # product revenue
    # c[13, 0] = -0.2 # purification cost
    # c[13, 2] = -0.2 # purification cost
    time_measurement = [4 * i for i in range(bn.n_time)]
    flag_estimate_reward_fun = True
    m = [-0.5] * 35 + [0]  # $/g
    b = np.transpose(np.array([[-0.5 / 1000] * 35]))  # $/g
    c = np.zeros(shape=(36, 5))
    c[35, 1] = 0.3  # product revenue
    c[35, 0] = -0.02  # purification cost
    c[35, 2] = -0.02  # purification cost
    gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
    theta = np.random.uniform(low=0.0, high=0.1, size=(bn.n_time - 1, bn.num_state, bn.num_action))
    grad = lambda x: - gradient.grad_f(x)
    f = lambda x: - gradient.func(x)


    def projection(y):
        theta = y.reshape((bn.n_time - 1, bn.num_state, bn.num_action))
        theta[:, 0, :] = np.clip(theta[:, 0, :], 0, 0.2)
        theta[:, 1, :] = np.clip(theta[:, 0, :], 0, 0.2)
        theta[0, 2, :] = np.clip(theta[0, 2, :], -0.1, 0.1)
        theta[:, 2, :] = np.clip(theta[:, 2, :], -0.2, 0.2)
        theta[:, 3, :] = np.clip(theta[:, 3, :], -0.1, 0.1)
        return theta.flatten()


    methods = {"PGD": cs.ProjectedGD(f, grad, projection,
                                     ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.))
               }

    max_iter = 3000
    tol = 1e-25
    for m_name in methods:
        print("\t", m_name)
        x = methods[m_name].solve(x0=theta.flatten(), max_iter=max_iter, tol=tol, disp=1)

    # # show convergence
    # fontsize = 24
    # figsize = (8, 6)
    # plt.figure(figsize=figsize)
    # for m_name in methods:
    #     plt.semilogy([f(x) for x in methods[m_name].get_convergence()], label=m_name)
    # plt.legend(fontsize=fontsize)
    # plt.xlabel("Number of iteration, $k$", fontsize=fontsize)
    # plt.ylabel(r"$f(x_k)$", fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # _ = plt.yticks(fontsize=fontsize)
    # plt.show()
    [f(x) for x in methods[m_name].get_convergence()]

    theta = x.reshape((bn.n_time - 1, bn.num_state, bn.num_action))
    policy = Policy(theta)
    transitions = []
    states = []
    current_transit = bn.initial_state_base[:-1] * bn.initial_state_base[
        -1]  # * bn.sample_sd[bn.num_action: bn.n_factor] #+  - bn.mu[bn.num_action: bn.n_factor]  # s- mu_s
    states.append(current_transit)
    for t in range(bn.n_time - 1):
        transitions.append(bn.beta_state[t].T + bn.beta_action[t].T @ theta[t].T)
        current_transit = transitions[-1] @ current_transit
        states.append(current_transit)
    reward = 0
    for t in range(bn.n_time):
        if t != bn.n_time - 1:
            a_t = bn.mu_a[t] + np.matmul(theta[t].T, states[t])
            reward += m[t] + np.dot(c[t], bn.rescale_state(states[t], t)) + np.dot(b[t], bn.rescale_action(a_t, t))
        else:
            reward += m[t] + np.dot(c[t], bn.rescale_state(states[t], t))

    bn.rescale_state(gradient.states[13], 13)
