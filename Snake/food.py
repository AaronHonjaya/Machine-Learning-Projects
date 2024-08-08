from typing import Tuple
import pygame
import game_utils as utils
import random


class Food:
    def __init__(self) -> None:
        self.food = pygame.Rect((650, 250, utils.GRID_SIZE, utils.GRID_SIZE))
    
    def respawn(self, head_position, body_positions) -> None:
        row = random.randint(0, utils.HEIGHT_IN_BLOCKS-1)
        column = random.randint(0, utils.WIDTH_IN_BLOCKS-1)
        while ([row, column] == head_position) or ([row, column] in body_positions):
            row = (row + 1) % utils.HEIGHT_IN_BLOCKS
            column = (column + 1) % utils.WIDTH_IN_BLOCKS
    
        self.food.x = column*utils.GRID_SIZE
        self.food.y = row*utils.GRID_SIZE
    
    def getBoardPos(self) -> Tuple[int, int]: 
        return utils.posToRowCol(self.food.topleft)
    
    def draw(self, screen) -> None:
        pygame.draw.rect(screen, utils.GREEN, self.food)