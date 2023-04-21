import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
import os
from pygame.math import Vector2
import gym_race

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.time = 0
        self.record = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.use_rays = False


    def load(self, model_name):
        pass


class Agent_DQN(Agent):

    def __init__(self):
        super().__init__()
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #-------------------------------------------------
        # MLP NETWORK ARCHITECTURE
        #-------------------------------------------------
        """
        # Model with distances to collisions
        self.input = 9
        self.layers = [self.input,18,18,3]
        self.use_rays = True
        """
        # Model with inmediate dangers
        self.input = 5
        self.layers = [self.input,256,3]
        self.use_rays = False
        self.model = Linear_QNet(self.layers)
        #-------------------------------------------------
        self.model_name = f'model{self.input}.pth' # self.input
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0.to(self.model.device)) # prediction by model (forward function)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
        """
        # Update the get_action method to accommodate the new 'BRAKE' action.
        self.epsilon = 80 - self.n_games
        final_move = np.zeros(4)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0.to(self.model.device))
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move
        """
        

    def load(self, model_name):
        # https://pythonguides.com/pytorch-load-model/
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, model_name)
        # self.model.load_state_dict(torch.load(file_name))

        checkPoint = torch.load(file_name)
        self.model.load_state_dict(checkPoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkPoint['optimizer_state_dict'])
        self.n_games = checkPoint['n_games']
        self.time = checkPoint['time']
        self.record = checkPoint['record']
        self.plot_scores = checkPoint['plot_scores']
        self.plot_mean_scores = checkPoint['plot_mean_scores']
        self.model.eval()

