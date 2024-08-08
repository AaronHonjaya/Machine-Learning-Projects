import math
import gym
from gym import spaces
import pygame
import torch
import game_utils as utils
import numpy as np
from snake import Snake
from food import Food



STATE_LEN = 13

class SnakeGameEnv():

    def __init__(self, render_frames = 30) -> None:
        pygame.init()
        self.block_size = utils.GRID_SIZE
        self.action_space = spaces.Discrete(3) 
        self.snake = Snake()
        self.food = Food()
        self.visited = np.zeros((utils.BOARD_SHAPE), dtype=np.uint32)
        self.visited = torch.tensor(self.visited, dtype=torch.uint8, device=utils.DEVICE)

        self.moves_made = 0
        self.score = 0
        self.render_frames = render_frames
        self.prev_dist = float('inf')

        
        self.screen = None
        self.clock = None
    
    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.moves_made = 0
        self.visited.zero_()

    
    def step(self, action):
        done = False
        self.moves_made += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                return None, None, None, done
        
        self.snake.updateDirWithAction(action)
        self.snake.move()
        
        
        
        ateFood = self.checkEatenFood()
        if(ateFood):
            self.food.respawn(self.snake.getHeadPosition(), self.snake.getBodyPositions())
            
        done = self.isGameOver() or self.moves_made > utils.WIDTH_IN_BLOCKS*utils.HEIGHT_IN_BLOCKS


        state = self.get_state()
        snake_head = self.snake.getHeadPosition()
        

        if done:
            reward = -10
        elif ateFood:
            self.visited.zero_()
            self.moves_made = 0
            reward = 10
        elif self.visited[snake_head[0]][snake_head[1]].item() > 1:
            reward = -1*self.visited[snake_head[0]][snake_head[1]].item()
        else:
            food_pos = self.food.getBoardPos()
            dist = math.sqrt((food_pos[0] - snake_head[0])**2 + (food_pos[1] - snake_head[1])**2)
            if dist >= self.prev_dist:
                reward = -1
            else:
                reward = 1
            self.prev_dist = dist
       


        if not done:
            # if self.visited[snake_head[0]][snake_head[1]] == 0:
            self.visited[snake_head[0]][snake_head[1]] += 1
            # self.visited[snake_head[0]][snake_head[1]]*=2

        return state, reward, self.score, done
    

    def get_state(self):
        surrounding_tiles = self._get_surrounding_tiles()
        body_pos = np.array(self.snake.getBodyPositions())
        head_pos = self.snake.getHeadPosition()
      
        
        res = np.zeros((STATE_LEN,))
        for i, pos in enumerate(surrounding_tiles):
            next_to_body = np.any(np.all(body_pos == pos, axis=1))
            outside_screen = pos[0] >= utils.HEIGHT_IN_BLOCKS or pos[1] >= utils.WIDTH_IN_BLOCKS or pos[0] < 0 or pos[1] < 0

            if next_to_body or outside_screen:
                res[i] = 1
        res[len(surrounding_tiles)+4:] = self._get_food_dir()
        # res[len(res)-1] = self.snake.length+1
        # if head_pos[0] == 0:
        #     print("hi")
        # print("head pos: ", head_pos, " | x,y pos : ", self.snake.head.topleft)
        # print("body positions: ", body_pos)
        # print("surrounding pos: ", surrounding_tiles)
        
        # print("res: ", res)


        return res
    
    def _get_food_dir(self):
        food_pos = self.food.getBoardPos()
        head_pos = self.snake.getHeadPosition()
        dir = np.zeros((4,))
        
        if food_pos[0] > head_pos[0]:
            dir[3] = 1  # Right
        elif food_pos[0] < head_pos[0]:
            dir[1] = 1  # Left

        if food_pos[1] > head_pos[1]:
            dir[2] = 1  # Down
        elif food_pos[1] < head_pos[1]:
            dir[0] = 1  # Up

        return dir


    def _get_surrounding_tiles(self):
        head_pos = np.array(self.snake.getHeadPosition())
        if self.snake.dir == utils.Direction.UP:
            straight_offset = [-1, 0]
            left_offset = [0, -1]
            right_offset = [0, 1]
        elif self.snake.dir == utils.Direction.DOWN:
            straight_offset = [1, 0]
            left_offset = [0, 1]
            right_offset = [0, -1] 
        elif self.snake.dir == utils.Direction.LEFT:
            straight_offset = [0, -1]
            left_offset = [1, 0]
            right_offset = [-1, 0]
        elif self.snake.dir == utils.Direction.RIGHT:
            straight_offset = [0, 1]
            left_offset = [-1, 0]
            right_offset = [1, 0]
        
        diag_right_offset = np.add(straight_offset, right_offset)
        diag_left_offset = np.add(straight_offset, left_offset)
        offsets = np.array([straight_offset, left_offset, right_offset,
                            diag_left_offset, diag_right_offset])  
        # offsets = np.array([straight_offset, left_offset, right_offset])   
        return offsets + head_pos

    def render(self):
        return self._render_frame()
    
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((utils.SCREEN_WIDTH, utils.SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        font = pygame.font.Font(None, 40) 
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 0)) 
        
        self.screen.fill(utils.BLACK)
        self.screen.blit(score_text, (10, 10))
        self.snake.draw(self.screen)
        self.food.draw(self.screen)
        self._draw_grid()
        self.clock.tick(self.render_frames)
        pygame.display.update()
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        
    def _draw_grid(self):
        for x in range(0, utils.SCREEN_WIDTH, utils.GRID_SIZE):
            for y in range(0, utils.SCREEN_HEIGHT, utils.GRID_SIZE):
                border = pygame.Rect((x, y, utils.GRID_SIZE, utils.GRID_SIZE))
                pygame.draw.rect(self.screen, utils.WHITE, border, 1)
    
    def isGameOver(self) -> bool:
        if (self.snake.head.centerx > utils.SCREEN_WIDTH or 
            self.snake.head.centerx < 0 or 
            self.snake.head.centery > utils.SCREEN_HEIGHT or
            self.snake.head.centery < 0):
                
                return True
        
        if (self.snake.length+1 >= utils.MAX_LENGTH):
            return True

        head_pos = self.snake.getHeadPosition()
        body_positions = self.snake.getBodyPositions()
        for pos in body_positions:
            if(pos == head_pos):
                return True
        return False

    def checkEatenFood(self) -> bool:
        head_col, head_row = utils.posToRowCol(self.snake.head.topleft)
        food_col, food_row = self.food.getBoardPos()
        if (head_col == food_col and head_row == food_row): 
            self.snake.grow()
            self.score += 1
            return True
        return False

    

    
