import numpy as np
import pygame
from game_utils import Direction
from snake import Snake
import game_utils as utils
from food import Food
from snakeGameEnv import STATE_LEN


def main():
    pygame.init()
    screen = pygame.display.set_mode((utils.SCREEN_WIDTH, utils.SCREEN_HEIGHT))

    snake = Snake(utils.GRID_SIZE)

    clock = pygame.time.Clock()
    fps = 60


    food = Food()

    run = True

    while run:
        clock.tick(fps)
        screen.fill(utils.BLACK)
        drawGrid(screen)
        snake.draw(screen)
        food.draw(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_RETURN):
                    print_state(snake, food)
                else:  
                    snake.updateDir(event.key)
        checkEatenFood(snake, food)
        
        pygame.display.update()


    pygame.quit()
    
def checkEatenFood(snake: Snake, food:Food) -> bool:
    head_col, head_row = utils.posToRowCol(snake.head.topleft)
    food_col, food_row = food.getBoardPos()
    if (head_col == food_col and head_row == food_row): 
        snake.grow()
        food.respawn(snake.getHeadPosition(), snake.getBodyPositions())
        return True
    return False

    
def drawGrid(screen):
    for x in range(0, utils.SCREEN_WIDTH, utils.GRID_SIZE):
        for y in range(0, utils.SCREEN_HEIGHT, utils.GRID_SIZE):
            border = pygame.Rect((x, y, utils.GRID_SIZE, utils.GRID_SIZE))
            pygame.draw.rect(screen, utils.WHITE, border, 1)


def print_state(snake: Snake, food:Food):
    surrounding_tiles = get_surrounding_tiles(snake, food)
    body_pos = np.array(snake.getBodyPositions())
    head_pos = snake.getHeadPosition()
    
    
    res = np.zeros((STATE_LEN,))
    for i, pos in enumerate(surrounding_tiles):
        next_to_body = np.any(np.all(body_pos == pos, axis=1))
        outside_screen = pos[0] >= utils.HEIGHT_IN_BLOCKS or pos[1] >= utils.WIDTH_IN_BLOCKS or pos[0] < 0 or pos[1] < 0

        if next_to_body or outside_screen:
            res[i] = 1
    res[len(surrounding_tiles)+4:] = get_food_dir(snake, food)
    
    print("head pos: ", head_pos, " | x,y pos : ", snake.head.topleft)
    print("body positions: ", body_pos)
    print("surrounding pos: ", surrounding_tiles)
    
    print("res: ", res)


    

def get_food_dir(snake: Snake, food:Food):
    food_pos = food.getBoardPos()
    head_pos = snake.getHeadPosition()
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


def get_surrounding_tiles(snake: Snake, food:Food):
    head_pos = np.array(snake.getHeadPosition())
    if snake.dir == utils.Direction.UP:
        straight_offset = [-1, 0]
        left_offset = [0, -1]
        right_offset = [0, 1]
    elif snake.dir == utils.Direction.DOWN:
        straight_offset = [1, 0]
        left_offset = [0, 1]
        right_offset = [0, -1] 
    elif snake.dir == utils.Direction.LEFT:
        straight_offset = [0, -1]
        left_offset = [1, 0]
        right_offset = [-1, 0]
    elif snake.dir == utils.Direction.RIGHT:
        straight_offset = [0, 1]
        left_offset = [-1, 0]
        right_offset = [1, 0]
    
    diag_right_offset = np.add(straight_offset, right_offset)
    diag_left_offset = np.add(straight_offset, left_offset)
    offsets = np.array([straight_offset, left_offset, right_offset,
                        diag_left_offset, diag_right_offset])  
    # offsets = np.array([straight_offset, left_offset, right_offset])   
    return offsets + head_pos

if __name__ == "__main__":
    main()