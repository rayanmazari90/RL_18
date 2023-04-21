

""" 
def simulate(agent, learning=True):  # LEARN
    total_reward = 0
    total_rewards = []

    env.set_view(True)

    for episode in range(NUM_EPISODES):

        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(.1)

                agent.model.save(agent.n_games,agent.time,record,plot_scores,plot_mean_scores,agent.trainer.optimizer,agent.model_name)

        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2  # continuous display of game

        for t in range(MAX_T):
            action = agent.get_action(state_0)
            obv, reward, done, info = env.step(action)
            state = state_to_bucket(obv)
            agent.remember(state_0, action, reward, state, done)
            total_reward += reward

            if learning:
                agent.train_short_memory(state_0, action, reward, state, done)
                agent.train_long_memory()

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.render(msgs=['SIMULATE',
                                 f'Episode: {episode}',
                                 f'Time steps: {t}',
                                 f'check: {info["check"]}',
                                 f'dist: {info["dist"]}',
                                 f'crash: {info["crash"]}',
                                 f'Reward: {total_reward:.0f}',
                                 ])

            if done or t >= MAX_T - 1:
                print("heloo")
                break

        agent.n_games = episode
        if learning:
            agent.model.save(agent.n_games,agent.trainer.optimizer,agent.model_name)
"""
import sys, os
import math, random
import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_race
from agents import Agent_DQN

VERSION_NAME = 'DQN_v02'  # the name for our model

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 100  # display live game every...


def simulate(agent, learning=True):  # LEARN
    total_rewards = []

    for episode in range(NUM_EPISODES):
        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2  # continuous display of game

        for t in range(MAX_T):
            state_old = state_0
            action = agent.get_action(state_old)
            obv, reward, done, info = env.step(action)
            state_new = state_to_bucket(obv)
            total_reward += reward

            if learning:
                agent.train_short_memory(state_old, action, reward, state_new, done)
                agent.remember(state_old, action, reward, state_new, done)

            state_0 = state_new

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                msgs=['SIMULATE',
                                 f'Episode: {episode}',
                                 f'Time steps: {t}',
                                 f'check: {info["check"]}',
                                 f'dist: {info["dist"]}',
                                 f'crash: {info["crash"]}',
                                 f'Reward: {total_reward:.0f}',
                                 ]
                print(msgs)
                env.render(msgs=msgs)

            if done or t >= MAX_T - 1:
                break

        agent.n_games = episode
        total_rewards.append(total_reward)

        if learning and episode % REPORT_EPISODES == 0:
            plt.plot(total_rewards)
            plt.ylabel('rewards')
            plt.show(block=False)
            plt.pause(.1)

        if learning:
            agent.train_long_memory()

            agent.model.save(agent.n_games, agent.time, agent.record, agent.plot_scores, agent.plot_mean_scores, agent.trainer.optimizer, agent.model_name)

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)

    return tuple(bucket_indice)

def load_and_play(agent, episode):
    agent.load_checkpoint(episode)
    simulate(agent, learning=False)


if __name__ == "__main__":
    env = gym.make("Pyrace-v1")
    agent = Agent_DQN()

    if not os.path.exists(f'models_{VERSION_NAME}'):
        os.makedirs(f'models_{VERSION_NAME}')

    if os.path.isfile('./model/'+agent.model_name):
        print('./model/'+agent.model_name)
        
    print('agent.model_name',agent.model_name)
    print(agent.model)
    print(agent)

    NUM_EPISODES = 100
    MAX_T = 2000

    NUM_BUCKETS  = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_ACTIONS  = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print(NUM_BUCKETS,NUM_ACTIONS,STATE_BOUNDS)

    

    simulate(agent, learning=True)
    # load_and_play(agent, 1000)  # Load a specific checkpoint and play without learning