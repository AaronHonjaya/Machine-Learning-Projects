import numpy as np
import random
import torch
from collections import deque
from snakeGameEnv import SnakeGameEnv, STATE_LEN
from model import LinearQModel, QTrainer
from tqdm import tqdm
import game_utils as utils

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001

class SnakeAgent:

    def __init__(self, env: SnakeGameEnv):
        self.env = env
        self.n_games = 0
        self.gamma = 0.9
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQModel(STATE_LEN, 128, 128, 3)
        self.model.to(utils.DEVICE)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        

    def fit(self, steps, render):
        record = 0
        with tqdm(total=steps) as pbar:
            for i in range(steps):
                old_state = self.env.get_state()
                action = self.get_action(old_state, isTrain=True)
                
                new_state, reward, score, done = self.env.step(action)
                
                self.train_short_memory(old_state, action, reward, new_state, done)
                self.remember(old_state, action, reward, new_state, done)

                if done:
                    self.env.reset()
                    self.n_games += 1
                    self.train_long_memory()
                    if score > record:
                        record = score
                        self.model.save("SnakeModel.pth")
                if render and self.n_games > 80:
                    self.env.render()
                
                pbar.set_postfix({
                    'Record' : record,
                    'Games': self.n_games,
                })
                pbar.update(1)
                # if self.n_games > 500:
                #     break
            
    def test(self, num_games, render):
        games_played = 1
        self.env.reset()
        self.model.load("SnakeModel.pth")
        while(games_played <= num_games):
            action = self.get_action(self.env.get_state(), isTrain= False)
            _new_state, _reward, score, done = self.env.step(action)
            
            if done:
                print(games_played, ".)   Score: ", score)
                games_played +=1
                self.env.reset()
            
            if render:
                self.env.render()




    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self,  state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, isTrain):
        self.epsilon = 80 - self.n_games
        if isTrain and random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
        else: 
            state0 = torch.tensor(state, dtype=torch.float, device= utils.DEVICE)
            move = torch.argmax(self.model(state0)).item()
        
        return move