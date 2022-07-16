
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from liboptpy.lib.mechanism_model import mechanism
from liboptpy.simulator.bio_env import fermentation_env2


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=40000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(16, activation="relu", kernel_initializer=last_init)(inputs)
    out = layers.Dropout(0.3)(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Dense(128, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(64, activation="relu")(state_input)
    # state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    # action_out = layers.Dense(32, activation="relu")(action_input)
    # action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_input])

    out = layers.Dense(8, activation="tanh")(concat)
    out = layers.Dropout(0.3)(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Dense(128, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


if __name__ == '__main__':
    macro_rep = 20
    avg_reward_lists = []
    for m in range(macro_rep):
        problem = "Feed-control-v0"
        process_model = mechanism()
        env = fermentation_env2(process_model)

        num_states = 7 # env.observation_space.shape[0]
        print("Size of State Space ->  {}".format(num_states))
        num_actions = 1 # env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(num_actions))

        upper_bound = 0.02 # upper bounad of feeding
        lower_bound = 0.0 # upper bounad of feeding

        def policy(state, noise_object):
            sampled_actions = tf.squeeze(actor_model(state))
            noise = noise_object()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise

            # We make sure action is within bounds
            legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

            return [np.squeeze(legal_action)]

        std_dev = 0.0001
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        actor_model = get_actor()
        critic_model = get_critic()

        target_actor = get_actor()
        target_critic = get_critic()

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.0001
        actor_lr = 0.0001

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        # Discount factor for future rewards
        gamma = 1
        # Used to update target networks
        tau = 0.00

        buffer = Buffer(50000, 64)


        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        total_episodes = 400

        avg_reward_best = 0

        # Takes about 20 min to train
        for ep in range(total_episodes):

            prev_state = env.reset()
            episodic_reward = 0
            length_ep = 0
            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy(tf_prev_state, ou_noise)
                # disturb training when it's stuck at 0 feeding rate
                if len(avg_reward_list) > 3 and abs(avg_reward_list[-2] - avg_reward_list[-1]) + abs(
                        avg_reward_list[-3] - avg_reward_list[-1]) < 1e-10:
                    action = [np.squeeze([0.003])]

                # Recieve state and reward from environment.
                # action[0] = 0.01
                state, reward, done = env.step(action)
                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward
                buffer.learn()
                update_target(tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            if avg_reward > avg_reward_best:
                avg_reward_best = avg_reward
                if avg_reward > 100:
                    file_path_actor = './results-runs/{0}/ddpg-actor-ep:{1}-alr:{2}-clr:{3}'.format(m, ep, actor_lr, critic_lr)
                    file_path_critic = './results-runs/{0}/ddpg-critic-ep:{1}-alr:{2}-clr:{3}'.format(m, ep, actor_lr, critic_lr)
                    actor_model.save(file_path_actor)
                    critic_model.save(file_path_critic)
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

        avg_reward_lists.append(avg_reward_list)

        file_path_actor = './results-runs/{0}/ddpg-actor-final-alr:{1}-clr:{2}'.format(m, actor_lr, critic_lr)
        file_path_critic = './results-runs/{0}/ddpg-critic-final-alr:{1}-clr:{2}'.format(m, actor_lr, critic_lr)
        actor_model.save(file_path_actor)
        critic_model.save(file_path_critic)
        np.save('avg_reward_lists.npy', avg_reward_lists)
        # Plotting graph
        # Episodes versus Avg. Rewards
        # plt.plot(avg_reward_list)
        # plt.xlabel("Episode")
        # plt.ylabel("Avg. Epsiodic Reward")
        # plt.show()


# episode_id = 378
# a = list(range(36))
# pd.DataFrame(data={"time":a,'action':buffer.action_buffer[(episode_id * 36) : ((episode_id+1)*36),:].flatten() * 1000})

