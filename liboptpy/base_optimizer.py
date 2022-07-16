import numpy as np
from collections import deque
import pandas as pd
from liboptpy.lib.mechanism_model import mechanism
from liboptpy.policy import Policy
from liboptpy.simulator.bio_env import fermentation_env2
from liboptpy.simulator.bioprocess import bioprocess_simulator


class LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, memory_size=1, patient=15, window=15, **kwargs):
        self.convergence = []
        self.convergence_f = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next)
        self._step_size = step_size
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)
        self.patience = patient
        self.window = window

        self.process_model = mechanism()
        self.env_eval = fermentation_env2(self.process_model)
        self.convergence_reward = []
        self.env = None
    def get_convergence(self):
        return self.convergence

    def evaluate(self, theta):
        x0 = self.env.bn.initial_state_full
        theta = theta.reshape((self.env.bn.n_time - 1, self.env.bn.num_state, self.env.bn.num_action))
        policy = Policy(theta, True)
        simulator = bioprocess_simulator(self.env.bn, self.env.time_measurement, policy, self.env.real_measurement)
        state = simulator.g(self.env.time_measurement, x0, self.env.ps)
        # feed_BN = pd.DataFrame(simulator.feed_every_second, columns=['time', 'feed_rate'], dtype="float64").drop_duplicates()
        # feed_BN = feed_BN.sort_values(by='time')
        feed_profile_ddpg = list(simulator.feed.values())
        total_feed_MDP = np.sum(feed_profile_ddpg)

        cumulative_reward = -0.13363 * 1000 * total_feed_MDP * 5 + 1.28739 * state[-1,1]

        return cumulative_reward

    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self.convergence_f = []
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
        self.convergence_f.append(self._f(self._x_current))
        iteration = 0

        while True:
            h = self.get_direction(self._x_current)
            self._grad_mem.append(h)
            self._alpha = self.get_stepsize()
            self._update_x_next()
            self._update_x_current()
            self._append_conv()
            iteration += 1
            if disp > 1:
                fx = self._f(self._x_current)
                self._append_conv_f(fx)
                # reward = self.evaluate(self._x_current)
                # self._append_conv_reward(reward)
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", fx)
                # print("Current reward =", reward)
                self._print_info()
            if self.check_convergence(tol):
                break
            if iteration >= max_iter:
                print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()
    
    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
        
    def _update_x_current(self):
        self._x_current = self._x_next
        
    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._x_current, self._alpha, self._grad_mem[-1])
        
    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._grad(self.convergence_f[-1]) - self._grad(self.convergence_f[-2]), ord=1) < tol
        
    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")
    
    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._grad(self._x_current))))
    
    def _append_conv(self):
        self.convergence.append(self._x_next)

    def _append_conv_f(self, fx):
        self.convergence_f.append(fx)

    def _append_conv_reward(self, reward):
        self.convergence_reward.append(reward)

    def _get_result_x(self):
        if self.patience > 0:
            min_index = np.argmax(self.convergence_f)
            return self.convergence[min_index]
        return self._x_current
    
class TrustRegionOptimizer(object):
    def __init__(self):
        raise NotImplementedError("Trust region methods are not implemented yet")