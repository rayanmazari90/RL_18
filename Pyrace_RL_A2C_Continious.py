import tensorflow as tf
import numpy as np
import gym
import os
import random
import matplotlib.pyplot as plt
from collections import deque
import math
import glob
import gym_race_continious
from gym.envs.registration import register
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

register(
    id='Pyrace-v2',
    entry_point='gym_race_continious.envs:RaceEnv',
    max_episode_steps=2000,
)

VERSION_NAME = 'A2C_FINAL_CONTINUS'
REPORT_EPISODES = 500
DISPLAY_EPISODES = 100


class A2CModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(A2CModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')

        self.actor_output = tf.keras.layers.Dense(num_actions)
        self.critic_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        actor = self.actor_output(x)
        critic = self.critic_output(x)
        return actor, critic


class Memory:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train_step(states, actions, rewards, next_states, dones, model, optimizer):
    with tf.GradientTape() as tape:
        actor_logits, _ = model(states)
        action_probs = tf.nn.softmax(actor_logits)
        action_log_probs = tf.nn.log_softmax(actor_logits)
        action_masks = tf.one_hot(actions, NUM_ACTIONS, dtype=tf.float32)
        selected_action_log_probs = tf.reduce_sum(action_masks * action_log_probs, axis=-1)

        _, next_value = model(next_states)
        _, value = model(states)
        next_value = tf.squeeze(next_value, axis=-1)
        value = tf.squeeze(value, axis=-1)

        td_targets = rewards + (1 - dones) * DISCOUNT_FACTOR * next_value
        td_errors = td_targets - value
        actor_loss = -tf.reduce_mean(selected_action_log_probs * td_errors)
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        loss = actor_loss + 0.5 * critic_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss.numpy()


def simulate(learning=True, model=None, start_episode=0):
    memory = Memory(MAX_T * 2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 64

    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = DISCOUNT_FACTOR
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 1000

    LEARNING = learning

    max_reward = -10_000

    env.set_view(True)

    for episode in range(start_episode, NUM_EPISODES):
        if episode > 0:
            total_rewards.append(total_reward)

            if LEARNING and episode % REPORT_EPISODES == 0:
                save_model(model, episode)
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(5)
                file = f'models_{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)
                file = f'models_{VERSION_NAME}/q_table_{episode}'
                print(f'models_{VERSION_NAME}/memory_{episode} saved')

        obv = env.reset()
        state_0 = obv
        total_reward = 0
        if not LEARNING:
            env.pyrace.mode = 2  # continuous display of game

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state_0, dtype=tf.float32), 0)
            actor_logits, _ = model(state_tensor)
            action_probs = tf.nn.softmax(actor_logits).numpy()
            if random.random() < explore_rate:
                action = env.action_space.sample()
            else:
                action = np.argmax(action_probs)

            obv, reward, done, info = env.step(action)
            state = obv
            env.remember(state_0, action, reward, state, done)
            total_reward += reward

            if LEARNING:
                memory.add(state_0, action, reward, state, done)
                if len(memory) >= batch_size:
                    states, actions, rewards, next_states, dones = zip(*memory.sample(batch_size))
                    states = np.array(states, dtype=np.float32)
                    actions = np.array(actions, dtype=np.int32)
                    rewards = np.array(rewards, dtype=np.float32)
                    next_states = np.array(next_states, dtype=np.float32)
                    dones = np.array(dones, dtype=np.float32)
                    loss = train_step(states, actions, rewards, next_states, dones, model, optimizer)

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.render(msgs=['SIMULATE',
                                    f'Episode: {episode}',
                                    f'Time steps: {t}',
                                    f'check: {info["check"]}',
                                    f'dist: {info["dist"]}',
                                    f'crash: {info["crash"]}',
                                    f'Reward: {total_reward:.0f}',
                                    f'Max Reward: {max_reward:.0f}'])

            if done or t >= MAX_T - 1:
                if total_reward > max_reward: max_reward = total_reward
                print("SIMULATE: Episode %d finished after %i time steps with total reward = %f."
                    % (episode, t, total_reward))
                break

        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def load_data(file):
    data = np.load(file, allow_pickle=True)
    print(type(data))
    print(data.shape)


    if data.shape[0] == 11:
        print('max min', data.max(), data.min(), 'total', data.sum())
        print('zeros', np.count_nonzero(data == 0), 'total', data.size)
    else:
        print('episodes', np.count_nonzero(data[:, 4] == True))
        return data


def save_model(model, episode):
    model.save_weights(f'models_{VERSION_NAME}/a2c_model_weights_{episode}.h5')
    print(f'models_{VERSION_NAME}/a2c_model_weights_{episode}.h5 saved')


def load_and_play(episode):
    a2c_model = A2CModel(NUM_ACTIONS)
    a2c_model.build((None,) + env.observation_space.shape)
    a2c_model.load_weights(f'models_{VERSION_NAME}/a2c_model_weights_{episode}.h5')
    print(f'models_{VERSION_NAME}/a2c_model_weights_{episode}.h5 loaded')

    # play game
    simulate(learning=False, model=a2c_model)


def get_last_saved_weights(directory):
    weight_files = glob.glob(f'{directory}/a2c_model_weights_*.h5')
    if not weight_files:
        return None, 0
    latest_weight_file = max(weight_files, key=os.path.getctime)
    last_episode = int(latest_weight_file.split("_")[-1].split(".")[0])
    return latest_weight_file, last_episode


if __name__ == "__main__":

    env = gym.make("Pyrace-v2")
    if not os.path.exists(f'models_{VERSION_NAME}'): os.makedirs(f'models_{VERSION_NAME}')

    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    print("action_space", env.action_space.n)
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print(NUM_BUCKETS, NUM_ACTIONS, STATE_BOUNDS)

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.8

    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print(DECAY_FACTOR)

    NUM_EPISODES = 65_000
    MAX_T = 2000

    SIMULATE = True

    a2c_model = A2CModel(NUM_ACTIONS)
    if SIMULATE:
        last_saved_weights, last_episode = get_last_saved_weights(f'models_{VERSION_NAME}')
        if last_saved_weights:
            print(f'Loading weights from: {last_saved_weights}')
            print("hello", last_episode)
            a2c_model.build((None,) + env.observation_space.shape)
            a2c_model.load_weights(last_saved_weights)
            simulate(model=a2c_model, start_episode=last_episode)
        else:
            simulate(model=a2c_model)

    else:
        last_saved_weights, last_episode = get_last_saved_weights(f'models_{VERSION_NAME}')
        if last_saved_weights:
            simulate(learning=False, model=a2c_model, start_episode=last_episode)
