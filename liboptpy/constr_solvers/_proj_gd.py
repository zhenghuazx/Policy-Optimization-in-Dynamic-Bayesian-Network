import numpy as np
import random

from liboptpy.lib.mechanism_model import mechanism
from liboptpy.simulator.bio_env import fermentation_env2
from ..base_optimizer import LineSearchOptimizer
from liboptpy.gradients.bayesian_network import bayesian_network, bayesian_network_posterior
from liboptpy.gradients.BN_gradient import BN_gradient
from liboptpy.policy import Policy
from liboptpy.lib.OUActionNoise import OUActionNoise

class ProjectedGD(LineSearchOptimizer):
    '''
    Class represents projected gradient method
    '''
    
    def __init__(self, f, grad, projector, step_size):
        super().__init__(f, grad, step_size)
        self._projector = projector
        
    def get_direction(self, x):
        return -self._grad(x)
    
    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)
    
    def check_convergence(self, tol):
        if self._f(self.convergence[-2]) - self._f(self.convergence[-1]) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class ProjectedBNPG(LineSearchOptimizer):
    '''
    Class represents projected gradient method
    '''

    def __init__(self, f, grad, projector, step_size, bn_posterior, env, sample_var, B=100, patient=15, window=15):
        super().__init__(f, grad, step_size, patient, window)
        self._projector = projector
        self.posterior = bn_posterior
        self.env = env
        self.B = B # bootstrapping
        self.sd = sample_var if len(sample_var.shape) == 1 else sample_var.to_numpy().flatten()

    def construct_bn_gradient(self, bn):
        time_interval = 140 / (bn.n_time -1)
        time_measurement = [time_interval * i for i in range(bn.n_time)]
        flag_estimate_reward_fun = False
        # m = [0] * (bn.n_time - 1) + [-15]  # $/g
        # b = np.transpose(np.array([[-0.1 * 1000 * time_interval] * (bn.n_time - 1)]))  # $/g
        # c = np.zeros(shape=(bn.n_time, bn.num_state))
        # c[bn.n_time - 1, 1] = 1.28739  # product revenue
        # c[bn.n_time - 1, 0] = 0  # purification cost
        # c[bn.n_time - 1, 2] = 0  # purification cost
        # gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
        gradient = BN_gradient(self.env.m, self.env.b, self.env.c, bn, time_measurement, flag_estimate_reward_fun)
        return gradient

    def get_direction(self, x):
        gradient = self.construct_bn_gradient(self.env.bn)
        grad = gradient.grad_f(x)
        return -grad

    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)

    def check_convergence(self, tol):
        if len(self.convergence_f) <= self.patience:
            return False
        if self.convergence_f[-1] > np.min(self.convergence_f[-self.window:]) and self.patience <= 0:
            return True
        else:
            self.patience -= 1
            return False
        # if self._f(self.convergence[-2]) - self._f(self.convergence[-1]) < tol:
        #     return True
        # else:
        #     return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print(
            "Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class ProjectedGDWithPosterior(LineSearchOptimizer):
    '''
    Class represents projected gradient method
    '''

    def __init__(self, f, grad, projector, step_size, bn_posterior, env, sample_var, B=100, patient=15, window=15):
        super().__init__(f, grad, step_size, patient, window)
        self._projector = projector
        self.posterior = bn_posterior
        self.env = env
        self.B = B # bootstrapping
        self.sd = sample_var if len(sample_var.shape) == 1 else sample_var.to_numpy().flatten()

    def construct_bn_gradient(self, bn):
        time_interval = 140 / (bn.n_time -1)
        time_measurement = [time_interval * i for i in range(bn.n_time)]
        flag_estimate_reward_fun = False
        # m = [0] * (bn.n_time - 1) + [-15]  # $/g
        # b = np.transpose(np.array([[-0.1 * 1000 * time_interval] * (bn.n_time - 1)]))  # $/g
        # c = np.zeros(shape=(bn.n_time, bn.num_state))
        # c[bn.n_time - 1, 1] = 1.28739  # product revenue
        # c[bn.n_time - 1, 0] = 0  # purification cost
        # c[bn.n_time - 1, 2] = 0  # purification cost
        # gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
        gradient = BN_gradient(self.env.m, self.env.b, self.env.c, bn, time_measurement, flag_estimate_reward_fun)
        return gradient

    def get_direction(self, x):
        p_beta, p_v2, mu = self.posterior.posterior_sample(self.B, useFixed=False)
        sum_bstrap_grad = np.zeros(shape=x.shape)
        for b in range(self.B):
            beta = p_beta[:,:,b] # np.apply_along_axis(np.mean, 2, p_beta)
            v2 = p_v2[:,b] # np.apply_along_axis(np.mean, 1, p_v2)
            bn = bayesian_network(mu, beta, v2, self.env.bn.num_action, self.env.bn.num_state, self.env.bn.n_time, True, mu, self.sd)
            gradient = self.construct_bn_gradient(bn)
            sum_bstrap_grad += gradient.grad_f(x)

        avg_grad = sum_bstrap_grad / self.B

        return -avg_grad

    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)

    def check_convergence(self, tol):
        if len(self.convergence_f) <= self.patience:
            return False
        if self.convergence_f[-1] > np.min(self.convergence_f[-self.window:]) and self.patience <= 0:
            return True
        else:
            self.patience -= 1
            return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print(
            "Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class ProjectedSGD(LineSearchOptimizer):
    '''
    Class represents projected stochastic gradient method
    '''

    def __init__(self, f, grad, projector, step_size, bn_posterior, sample_var, env, batch_size=8, B=16, patient=15, window=15):
        super().__init__(f, grad, step_size, patient, window)
        self.grad = grad
        self._projector = projector
        self.env = env
        self.posterior = bn_posterior
        self.batch_size = batch_size
        self.sd = sample_var
        self.B = B

    def _yield_batches(self, x):
        theta = x.reshape((self.posterior.n_time - 1, self.posterior.num_state, self.posterior.num_action))
        std_dev = 0.005
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        policy = Policy(theta, True, ou_noise)
        return self.env.generate_trajectories(policy, self.batch_size)

    def construct_bn_gradient(self, bn):
        time_interval = 140 / (bn.n_time -1)
        time_measurement = [time_interval * i for i in range(bn.n_time)]
        flag_estimate_reward_fun = False
        # m = [0] * (bn.n_time - 1) + [-15]  # $/g
        # b = np.transpose(np.array([[-0.1 * 1000 * time_interval] * (bn.n_time - 1)]))  # $/g
        # c = np.zeros(shape=(bn.n_time, bn.num_state))
        # c[bn.n_time - 1, 1] = 1.28739  # product revenue
        # c[bn.n_time - 1, 0] = 0  # purification cost
        # c[bn.n_time - 1, 2] = 0  # purification cost
        # gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
        gradient = BN_gradient(self.env.m, self.env.b, self.env.c, bn, time_measurement, flag_estimate_reward_fun)
        return gradient

    def get_direction(self, x):
        data, total_rewards = self._yield_batches(x)
        p_beta, p_v2, mu = self.posterior.posterior_sample(self.B, useFixed=False)
        norm_data = (data - mu) / self.sd
        sum_bstrap_grad = np.zeros(shape=x.shape)
        for b in range(self.B):
            beta = p_beta[:,:,b] # np.apply_along_axis(np.mean, 2, p_beta)
            v2 = p_v2[:,b] # np.apply_along_axis(np.mean, 1, p_v2)
            bn = bayesian_network(mu, beta, v2, self.env.bn.num_action, self.env.bn.num_state, self.env.bn.n_time, True, mu, self.sd)
            gradient = self.construct_bn_gradient(bn)
            grad = gradient.grad_sgd(x, norm_data, total_rewards)
            sum_bstrap_grad += grad

        avg_grad = sum_bstrap_grad / self.B

        return -avg_grad

        # return -self._grad(x, batch)

    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)

    def check_convergence(self, tol):
        if len(self.convergence_f) <= self.patience:
            return False
        if self.convergence_f[-1] > np.min(self.convergence_f[-self.window:]) and self.patience <= 0:
            return True
        else:
            self.patience -= 1
            return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class ProjectedBNSGD(LineSearchOptimizer):
    '''
    Class represents projected gradient method with point estimator
    '''

    def __init__(self, f, grad, projector, step_size, bn_posterior, sample_var, env, batch_size=8, B=50, patient=15, window=15):
        super().__init__(f, grad, step_size, patient, window)
        self._projector = projector
        self.env = env
        self.posterior = bn_posterior
        self.batch_size = batch_size
        self.sd = sample_var
        self.B = B

    def _yield_batches(self, x):
        theta = x.reshape((self.posterior.n_time - 1, self.posterior.num_state, self.posterior.num_action))
        std_dev = 0.005
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        policy = Policy(theta, True, ou_noise)
        return self.env.generate_trajectories(policy, self.batch_size)

    def construct_bn_gradient(self, bn):
        time_interval = 140 / (bn.n_time -1)
        time_measurement = [time_interval * i for i in range(bn.n_time)]
        flag_estimate_reward_fun = False
        # m = [0] * (bn.n_time - 1) + [-15]  # $/g
        # b = np.transpose(np.array([[-0.1 * 1000 * time_interval] * (bn.n_time - 1)]))  # $/g
        # c = np.zeros(shape=(bn.n_time, bn.num_state))
        # c[bn.n_time - 1, 1] = 1.28739  # product revenue
        # c[bn.n_time - 1, 0] = 0  # purification cost
        # c[bn.n_time - 1, 2] = 0  # purification cost
        # gradient = BN_gradient(m, b, c, bn, time_measurement, flag_estimate_reward_fun)
        gradient = BN_gradient(self.env.m, self.env.b, self.env.c, bn, time_measurement, flag_estimate_reward_fun)
        return gradient

    def get_direction(self, x):
        data, total_rewards = self._yield_batches(x)
        p_beta, p_v2, mu = self.posterior.posterior_sample(self.B, useFixed=True)
        norm_data = (data - mu) / self.sd

        gradient = self.construct_bn_gradient(self.env.bn)
        grad = gradient.grad_sgd(x, norm_data, total_rewards)

        return -grad

        # return -self._grad(x, batch)

    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)

    def check_convergence(self, tol):
        if len(self.convergence_f) <= self.patience:
            return False
        if self.convergence_f[-1] > np.min(self.convergence_f[-self.window:]) and self.patience <= 0:
            return True
        else:
            self.patience -= 1
            return False
        # if self._f(self.convergence_f[-2]) - self._f(self.convergence_f[-1]) < tol:
        #     return True
        # else:
        #     return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))