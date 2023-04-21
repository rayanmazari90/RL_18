import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
# from torch.autograd import Variable

# Define model
class Linear_QNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # super(Linear_QNet, self).__init__()
        self.layers = nn.ModuleList([])
        for k in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[k],layers[k+1]))



        # Making the code device-agnostic
        self.device = 'cpu'
        # Instantiating a pre-trained model
        # model = models.resnet18(pretrained=True)
        # Transferring the model to a CUDA enabled GPU
        self = self.to(self.device)
        # Now the reader can continue the rest of the workflow
        # including training, cross validation, etc!


    def forward(self, x):
        for k in range(len(self.layers)-1):
            x = F.relu(self.layers[k](x)) # F.sigmoid
        x = self.layers[-1](x)
        return x

    def save(self, n_games, time, record, plot_scores, plot_mean_scores, optimizer, model_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, model_name)
        # torch.save(self.state_dict(), file_name)
        torch.save({
            'n_games': n_games,
            'time': time,
            'record': record,
            'plot_scores': plot_scores,
            'plot_mean_scores': plot_mean_scores,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        self.criterion = nn.MSELoss()
        self.device = 'cpu'

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)
        # (n, x) # batch samples

        if len(state.shape) == 1: # in case of single sample...
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # pred.clone()
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # BELLMAN EQUATION (SIMPLIFIED VERSION)
                # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
                # preds[argmax(action)] = Q_new
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad() # EMPTY THE GRADIENTS
        loss = self.criterion(target, pred)
        loss.backward() # BACKPROPAGATION

        self.optimizer.step()
