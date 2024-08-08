import gym
from gym import spaces
import pygame
import game_utils as utils
import numpy as np
from snake import Snake
from food import Food

class SnakeGameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(self, render_mode = None) -> None:
        pygame.init()
        self.block_size = utils.GRID_SIZE
        self.action_space = spaces.Discrete(3) 
        self.snake = Snake()
        self.food = Food()
        self.visited = np.zeros((utils.BOARD_SHAPE), dtype=bool)
        self.moves_made = 0
        self.score = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.uint8)
    
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
    
    def reset(self):
        self.snake = Snake()
        self.food = Food()
        return self._get_observation()
    
    def step(self, action):
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                return None, None, done, {}
        
        self.snake.updateDirWithAction(action)
        self.snake.move()
        if(self.checkEatenFood()):
            self.food.respawn()

        done = self.isGameOver() or  self.moves_made > 100*(self.snake.length+1)
        obs = self._get_observation()
        ateFood = self.snake.checkEatenFood(self.food)
        
        snake_head = self.snake.getHeadPosition()

        if done:
            reward = -50
        elif ateFood:
            self.visited = np.zeros((BOARD_SHAPE), dtype=bool)
            reward = 50
        # elif self.visited[snake_head[0]][snake_head[1]]:
        #     reward = -5
        # elif num_body_surrounding_head == 2:
        #     reward = -50
        else:
            reward = 0


        if not done:
            self.visited[snake_head[0]][snake_head[1]] = True


        return obs, reward, done, False, {}
    

    def _get_observation(self):
        options = self._get_surrounding_tiles()
        body_pos = self.snake.getBodyPositions()


        res = np.zeros(self.observation_space.shape)
        num_surrounding_body=0
        for i, pos in enumerate(options):
            next_to_body = pos.tolist() in body_pos
            if ( next_to_body or pos[0] >= utils.HEIGHT_IN_BLOCKS 
                or pos[1] >= utils.WIDTH_IN_BLOCKS or pos[0] < 0 or pos[1] < 0):
                    res[i] = 1
            if next_to_body:
                num_surrounding_body += 1

        res[len(options)+self.snake.dir] = 1

        
        res[len(options)+4:] = self._get_food_dir()


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
        head_pos = self.snake.getHeadPosition()
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
        
        return [np.array(head_pos) + np.array(straight_offset),
                np.array(head_pos)+np.array(left_offset), 
                np.array(head_pos)+np.array(right_offset)]


    def render(self, mode):
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
        self.clock.tick(SnakeGameEnv.metadata["render_fps"])
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

    

    
