import numpy as np
import tensorflow as tf

class PolicyDDPG:
    def __init__(self, actor_model, lower_bound, upper_bound, scale_needed=False, policy_algo='DDPG'):
        self.actor_model = actor_model
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scale_needed = scale_needed
        self.policy_algo = policy_algo

    def next_action(self, state, t):
        state_full = np.concatenate([state, np.array([t])])
        state_full = state_full.reshape(1,len(state_full))
        sampled_actions = tf.squeeze(self.actor_model.predict(state_full))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy()
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]


class Policy:
    def __init__(self, theta, scale_needed, noise_object = None, policy_algo='BN-MDP'):
        self.scale_needed = scale_needed
        self.theta = theta
        self.noise_object = noise_object
        self.policy_algo = policy_algo

    def next_action(self, state, t):
        if t == 0 or t == 35:
            return 0
        else:
            if self.noise_object is not None:
                noise = self.noise_object()
                return self.theta[t].T @ state + noise
            else:
                return self.theta[t].T @ state



class PolicyLab:
    def __init__(self, time, measurement, scale_needed, policy_algo=None):
        self.times = np.array(time)
        self.scale_needed = scale_needed
        self.measurement = measurement
        self.policy_algo = policy_algo

    def next_action(self, state, t):
        if t < 0:
            return self.measurement[t]
        index = 0
        time_w = -1
        for i, time in enumerate(self.times):
            if time - t > 0:
                index = i - 1
                break
        # index = abs(t - self.time + 0.5).argmin()
        return self.measurement[index]
