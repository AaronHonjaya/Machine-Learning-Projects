from agent import SnakeAgent
from snakeGameEnv import SnakeGameEnv
import game_utils as utils
import torch
import numpy as np

def main():
    print(utils.DEVICE)

    env = SnakeGameEnv(render_frames=30)
    agent = SnakeAgent(env)
    agent.fit(20000, True)
    agent.test(5, True) 
    
if __name__ == "__main__":
    main()