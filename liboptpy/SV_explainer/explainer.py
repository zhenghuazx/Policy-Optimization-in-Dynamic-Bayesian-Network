import numpy as np
import matplotlib.pyplot as plt
from liboptpy.gradients.bayesian_network import bayesian_network, bayesian_network_posterior


class Explainer:
    def __init__(self, bn, policy, posterior=None):
        self.bn = bn
        self.theta = policy.theta if len(policy.theta.shape) == 3 else policy.theta.reshape(
            (self.bn.n_time - 1, self.bn.num_state, self.bn.num_action))
        self.B=5000
        self.posterior = posterior
    def get_average_shapley_value(self, current_state, current_action, cur_time, target_time):
        p_beta, p_v2, mu = self.posterior.posterior_sample(self.B, useFixed=False)
        sv = []
        for b in range(self.B):
            beta = p_beta[:,:,b] # np.apply_along_axis(np.mean, 2, p_beta)
            v2 = p_v2[:,b] # np.apply_along_axis(np.mean, 1, p_v2)
            bn_b = bayesian_network(mu, beta, v2, bn.num_action, bn.num_state, bn.n_time, True, mu, self.bn.sample_sd)
            sv.append(self.get_shapley_value(current_state, current_action, cur_time, target_time, bn_b))
        return sum(sv) / self.B

    def get_shapley_value(self, current_state, current_action, cur_time, target_time, bn_b = None):
        bn = self.bn if bn_b is None else bn_b

        mean_action = bn.sample_mean[(bn.n_factor *  cur_time):(bn.n_factor *  cur_time + bn.num_action)]
        sd_action = bn.sample_sd[(bn.n_factor * cur_time):(bn.n_factor * cur_time + bn.num_action)]
        mean_state = bn.sample_mean[(bn.n_factor * (cur_time+1) - bn.num_state):(bn.n_factor * (cur_time+1))]
        sd_state = bn.sample_sd[(bn.n_factor * (cur_time+1) - bn.num_state):(bn.n_factor * (cur_time+1))]

        mean_state_target = bn.sample_mean[(bn.n_factor * (target_time+1) - bn.num_state):(bn.n_factor * (target_time+1))]

        current_norm_action = (current_action - mean_action) # / sd_action
        current_transit = (current_state - mean_state) # / sd_state

        shapley_values = []

        for i in range(bn.num_action):
            temp = [0] * bn.num_action
            temp[i] = current_norm_action[i]
            out_sv = self._get_shapley_value_for_each_action(cur_time, target_time, temp, bn)
            target_CQAs = np.array(out_sv[-1].flatten())
            shapley_values.append(bn.rescale_state(target_CQAs, target_time) - mean_state_target)
            # shapley_values.append(target_CQAs)

        for i in range(bn.num_state):
            temp = [0] * bn.num_state
            temp[i] = current_transit[i]
            out_sv = self._get_shapley_value_for_each_state(cur_time, target_time, temp, bn)
            target_CQAs = np.array(out_sv[-1])
            shapley_values.append(bn.rescale_state(target_CQAs, target_time) - mean_state_target)
            # shapley_values.append(target_CQAs)
        return np.array(shapley_values) #row: input state; col: output state


    def _get_shapley_value_for_each_state(self, cur_time, target_time, current_transit, bn):
        transitions = []
        states = []
        states.append(current_transit)
        self.states = states
        for t in range(cur_time, target_time - 1):
            transitions.append(bn.beta_state[t].T + bn.beta_action[t].T @ self.theta[t].T)
            current_transit = transitions[-1] @ current_transit
            states.append(current_transit)

        return states

    def _get_shapley_value_for_each_action(self, cur_time, target_time, current_action, bn):
        transitions = []
        states = []
        current_transit = bn.beta_action[cur_time].T @ np.array(current_action).reshape(bn.num_action, 1)
        states.append(current_transit)
        self.states = states
        for t in range(cur_time + 1, target_time - 1): # bn.n_time - 1
            transitions.append(bn.beta_state[t].T + self.bn.beta_action[t].T @ self.theta[t].T)
            current_transit = transitions[-1] @ current_transit
            states.append(current_transit)

        return states

if __name__ == '__main__':
    import xgboost
    import shap
    import pandas as pd
    from liboptpy.SV_explainer.waterfall_plot import waterfall
    # feed rate, X_f, C, S, N, V

    beta_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/beta_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    v2_gibbs = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/v2_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None,
                            dtype=np.float64)
    the_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/the_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    tau2_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/tau2_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    kap_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/kap_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    lam_R = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/lam_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    mu = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/mu_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)
    sd = pd.read_csv('/Users/hua.zheng/Research/PHD/project/BN-MDP/BN/sd_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma3.txt', header=None, dtype=np.float64)

    bn_post = bayesian_network_posterior(the_R.to_numpy(), tau2_R.to_numpy(), kap_R.to_numpy(), lam_R.to_numpy(),
                                         mu.to_numpy().flatten(), 5, 1, 36, beta_gibbs, v2_gibbs)
    p_beta, p_v2, mu = bn_post.posterior_sample(5000, useFixed=False)
    beta = np.apply_along_axis(np.mean, 2, p_beta)
    v2 = np.apply_along_axis(np.mean, 1, p_v2)
    bn = bayesian_network(mu, beta, v2, 1, 5, 36, True, mu, sd.to_numpy().flatten())


    '''
    Single Bayesian network
    '''
    target_CQA= 1
    init_states = [37.5, 13.6, 26.5, 1.97, 0.674]
    init_action = [0.008]

    # init_states = [35.8, 16.3, 13.6, 1.29, 0.731]
    # init_action = [0.00362]
    # init_states = [19.4, 10.1, 3.18, 2.38, 0.660]
    # init_action = [0.00730]

    target_time = 35
    cur_time = 15
    explainer = Explainer(bn, policy)
    final = explainer.get_shapley_value(init_states, init_action, cur_time, target_time)


    mean_final = bn.sample_mean[(bn.n_factor * (target_time+1) - bn.num_state):(bn.n_factor * (target_time+1))]
    # CPP + CQA
    shap_values_1 = shap.Explanation(final[:,target_CQA], base_values=mean_final[target_CQA], data=init_action + list(init_states), feature_names=['feed rate', 'X_f', 'C', 'S', 'N', 'V'])
    waterfall(shap_values_1, max_display=6)

    # only CQA
    shap_values_1 = shap.Explanation(final[1:,target_CQA], base_values=mean_final[target_CQA], data=list(init_states), feature_names=['X_f', 'C', 'S', 'N', 'V'])
    waterfall(shap_values_1, max_display=6)

    g = shap.plots.bar(shap_values_1, max_display=11,show=False)
    plt.xlabel('SV Feature Importantce', fontsize=13)
    plt.show()

    '''
    Posterior distribution of Bayesian network
    '''
    target_CQA = 1
    init_states = [37.5, 13.6, 26.5, 1.97, 0.674]
    init_action = [0.008]

    # init_states = [35.8, 16.3, 13.6, 1.29, 0.731]
    # init_action = [0.00362]
    # init_states = [19.4, 10.1, 3.18, 2.38, 0.660]
    # init_action = [0.00730]

    target_time = 35
    cur_time = 15
    explainer = Explainer(bn, policy, bn_post)
    final = explainer.get_average_shapley_value(init_states, init_action, cur_time, target_time)

    mean_final = bn.sample_mean[(bn.n_factor * (target_time+1) - bn.num_state):(bn.n_factor * (target_time+1))]
    # # CPP + CQA
    # shap_values_1 = shap.Explanation(final[:,target_CQA], base_values=mean_final[target_CQA], data=init_action + list(init_states), feature_names=['feed rate', 'X_f', 'C', 'S', 'N', 'V'])
    # shap.plots.waterfall(shap_values_1, max_display=6)

    # only CQA
    shap_values_1 = shap.Explanation(final[1:,target_CQA], base_values=mean_final[target_CQA], data=list(init_states), feature_names=['X_f', 'C', 'S', 'N', 'V'])
    waterfall(shap_values_1, 6, True, "\n$E[C_H]$", "$E[C_H|\mathcal{O}_t]$")

    g = shap.plots.bar(shap_values_1, max_display=11,show=False)
    plt.xlabel('SV Feature Importantce', fontsize=13)
    plt.show()