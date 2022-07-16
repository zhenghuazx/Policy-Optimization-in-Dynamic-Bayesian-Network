import numpy as np
import pandas as pd

class bayesian_network_posterior:
    def __init__(self, the_R, tau2_R, kap_R, lam_R, mu, num_state, num_action, n_time, beta, v2):
        self.the_R = the_R
        self.tau2_R = tau2_R
        self.kap_R = kap_R
        self.lam_R = lam_R
        self.num_nodes = (num_state + num_action) * n_time
        self.num_state = num_state
        self.num_action = num_action
        self.mu = mu
        self.beta = beta
        self.v2 = v2
        self.n_time = n_time

    def posterior_sample(self, size=1, useFixed = True):
        p_beta = np.zeros(shape=(self.num_nodes,self.num_nodes, size))
        p_v2 = np.zeros(shape=(self.num_nodes, size))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.tau2_R[i, j] != 0:
                    p_beta[i, j, ] = np.random.normal(loc=self.the_R[i,j], scale=np.sqrt(self.tau2_R[i,j]), size=size)
            gamma_rate = self.lam_R[i] / 2
            p_v2[i,] = 1 / np.random.gamma(shape=self.kap_R[i] / 2, scale=1/gamma_rate, size=size)
        if useFixed:
            p_beta = self.beta
            p_v2 = self.v2
        return (p_beta, p_v2, self.mu)
        # use np.apply_along_axis(np.mean,2,p_beta)

class bayesian_network:
    def __init__(self, mu, beta, v2, num_action, num_state, n_time, normalized = True, sample_mean=None, sample_sd=None):
        self.n_time = n_time
        if normalized:
            self.sample_mean = sample_mean
            self.sample_sd = sample_sd
            self.normalized = True
        self.initial_state_full = np.array([0.05, 0.00, 0.00, 30.00, 5.00,0.7])
        if num_state == 4:
            self.initial_state_base = self.initial_state_full[[0, 1, 3, 4, 5]]
        if num_state == 5:
            self.initial_state_base = self.initial_state_full
        self.mu = mu
        self.v2 = v2
        self.beta = beta
        self.num_action = num_action
        self.num_state = num_state
        self.n_factor = num_action + num_state
        self.beta_state = np.zeros(shape=(n_time, num_state, num_state)) # s -> s
        self.beta_action = np.zeros(shape=(n_time, num_action, num_state)) # a -> s
        for i in range(n_time-1):
            self.beta_state[i,:,:] = beta[(self.n_factor * (i+1) - num_state):(self.n_factor * (i+1)), (self.n_factor * (i + 2) - num_state): (self.n_factor * (i + 2))]
            self.beta_action[i,:,:] = beta[(self.n_factor * i):(self.n_factor * i + self.num_action), (self.n_factor * (i + 2) - num_state):(self.n_factor * (i + 2))]
        self.mu_a = []
        for i in range(self.n_time):
            temp_list = []
            for j in range(self.num_action):
                temp_list.append(self.mu[i * self.n_factor + j])
            self.mu_a.append(temp_list)

    def initial_state_generator(self, scale=10):
        init_states = self.initial_state_base + np.abs(np.random.normal(0, np.array(self.initial_state_base)/scale + 0.01))
        init_states = init_states[[0,1,3,4,5]] # init_states[:-1] * init_states[-1]
        self.initial_state = (init_states - self.sample_mean[(self.n_factor - self.num_state):self.n_factor]) / self.sample_sd[(self.n_factor - self.num_state):self.n_factor]

    def rescale_action(self, action, t, scale_method = "standard"):
        if not self.normalized:
            return action
        if scale_method == "standard":
            return self.sample_sd[(self.n_factor * t):(self.n_factor * t + self.num_action)] * action + self.sample_mean[(self.n_factor * t):(self.n_factor * t + self.num_action)]
        else:
            return self.sample_sd[(self.n_factor * t):(self.n_factor * t + self.num_action)] * action

    def rescale_state(self, state, t, scale_method = "standard"):
        if not self.normalized:
            return state
        if scale_method == "standard":
            return self.sample_sd[(self.n_factor * (t+1) - self.num_state):(self.n_factor * (t+1))] * state + self.sample_mean[(self.n_factor * (t+1) - self.num_state):(self.n_factor * (t+1))]
        else:
            return self.sample_sd[(self.n_factor * (t+1) - self.num_state):(self.n_factor * (t+1))] * state



if __name__ == '__main__':
    the_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/the_R.txt', header=None, dtype=np.float64)
    tau2_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/tau2_R.txt', header=None, dtype=np.float64)
    kap_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/kap_R.txt', header=None, dtype=np.float64)
    lam_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/lam_R.txt', header=None, dtype=np.float64)
    mu = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/mu.txt', header=None, dtype=np.float64)
    sd = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/sd.txt', header=None, dtype=np.float64)
    bn_post = bayesian_network_posterior(the_R.to_numpy(), tau2_R.to_numpy(), kap_R.to_numpy(), lam_R.to_numpy(), mu.to_numpy().flatten(), 5, 2, 14)
    p_beta, p_v2, mu = bn_post.posterior_sample(2)
    beta = np.apply_along_axis(np.mean, 2, p_beta)
    v2 = np.apply_along_axis(np.mean, 1, p_v2)
    bn = bayesian_network(mu,beta,v2, 2,5,14, True, mu, sd.to_numpy().flatten())

    beta_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/beta_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10-2operations.txt', header=None, dtype=np.float64)
    v2_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/v2_s5a1-R100-explore0.3-v1-modelrisk--ntime36-sigma10.txt', header=None,
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
