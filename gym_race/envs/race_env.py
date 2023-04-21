import gym
from gym import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int)
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []

    def reset(self):
        mode = self.pyrace.mode
        del self.pyrace
        self.pyrace = PyRace2D(self.is_view, mode = mode)
        obs = self.pyrace.observe()
        return obs

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return obs, reward, done, {'dist':self.pyrace.car.distance, 'check':self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive}

    def render(self, mode="human", close=False, msgs=[]):
        if self.is_view:
            self.pyrace.view_(msgs)

    def set_view(self, flag):
        self.is_view = flag

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
