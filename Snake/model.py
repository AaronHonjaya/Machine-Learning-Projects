import os
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch import optim
import numpy as np
import game_utils as utils



class LinearQModel(nn.Module):
    
    
    def __init__(self, in_size, h1_size, h2_size, out_size) -> None:
        super().__init__()
        self.input = nn.Linear(in_size, h1_size)
        self.h1 = nn.Linear(h1_size, h2_size)
        self.out = nn.Linear(h2_size, out_size)

    def forward(self, x):
        x = relu(self.input(x))
        x = relu(self.h1(x))
        return self.out(x)
    
    def save(self, file_name = "model.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name="model.pth"):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        try:
            self.load_state_dict(torch.load(file_name))
            print(f"Model loaded from {file_name}")
        except Exception as e:
            print(f"Error loading the model: {e}")


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float, device= utils.DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device= utils.DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long, device= utils.DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device= utils.DEVICE)
        # state = torch.tensor(np.array(state), dtype=torch.float)
        # next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        # action = torch.tensor(np.array(action), dtype=torch.long)
        # reward = torch.tensor(np.array(reward), dtype=torch.float)

        if (len(state.shape) == 1):
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            # done = (done, )
            done = torch.tensor((done,), dtype=torch.float, device=utils.DEVICE)
        else:
            done = torch.tensor(done, dtype=torch.float, device=utils.DEVICE)
            
        # print(action)
        pred = self.model(state)
            
        next_pred = self.model(next_state)

        Q_new_vec = reward + self.gamma * torch.max(next_pred, dim=1)[0] * (1 -done)
         
        target = pred.clone()
        target[range(len(action)), action] = Q_new_vec
        
        # target1 = pred.clone()

        # for i in range(len(done)):
        #     Q_new = reward[i]
        #     if not done[i]:
        #         Q_new = reward[i]+self.gamma*torch.max(self.model(next_state[i]))
            
        #     target1[i][action[i].item()] = Q_new
        
        # target = torch.tensor(target, dtype=torch.float32, device=utils.DEVICE)
        # target1 = torch.tensor(target1, dtype=torch.float32, device=utils.DEVICE)
        # if not torch.allclose(target, target1, atol=1e-5, rtol=1e-3):
        #     print("failed")
        
        self.optim.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optim.step()

