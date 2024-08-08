from typing import List, Tuple
import pygame
import game_utils as utils
from game_utils import Action, Direction
from food import Food

class Snake:
    def __init__(self, grid_size = utils.GRID_SIZE):
        self.length = 4
        self.dir = Direction.RIGHT
        self.block_size = grid_size
        self.head = pygame.Rect((300, 250, grid_size, grid_size))

        self.body = []
        self.initBody(300, 250)
        
        self.move_event = pygame.USEREVENT + 1
        pygame.time.set_timer(self.move_event, 500)
        self.allow_dir_change = True 

        self.skip_tail_update = False
    
    def initBody(self, start_x, start_y) -> None:
        for i in range(1, self.length):
            self.body.append(pygame.Rect((start_x - i*self.block_size, start_y, 
                                          self.block_size, self.block_size)))


    def move(self) -> None:
        prevx = self.head.x
        prevy = self.head.y

        if self.dir == Direction.LEFT:
            self.head.move_ip(-self.block_size,0)
        elif self.dir == Direction.RIGHT:
            self.head.move_ip(self.block_size,0)
        elif self.dir == Direction.UP:
            self.head.move_ip(0, -self.block_size)
        elif self.dir == Direction.DOWN:
            self.head.move_ip(0, self.block_size)
        
        
        for i, block in enumerate(self.body):
            if self.skip_tail_update and i == len(self.body) - 1:
                self.skip_tail_update = False
                continue
            tempx = block.x
            tempy = block.y
            block.x = prevx
            block.y = prevy
            prevx = tempx
            prevy = tempy
        
        self.allow_dir_change = True
    
    def updateDir(self, key_pressed) -> None:
        if self.allow_dir_change:
            if key_pressed == pygame.K_UP:
                if self.dir != Direction.DOWN:
                    self.dir = Direction.UP
                    self.allow_dir_change = False
                    self.move()
            elif key_pressed == pygame.K_DOWN:
                if self.dir != Direction.UP:
                    self.dir = Direction.DOWN
                    self.allow_dir_change = False
                    self.move()
            elif key_pressed == pygame.K_LEFT:
                if self.dir != Direction.RIGHT:
                    self.dir = Direction.LEFT
                    self.allow_dir_change = False
                    self.move()
            elif key_pressed == pygame.K_RIGHT:
                if self.dir != Direction.LEFT:
                    self.dir = Direction.RIGHT
                    self.allow_dir_change = False
                    self.move()
        

    def updateDirWithAction(self, action) -> None:
        if action == Action.NONE:
            return
        elif action == Action.LEFT:
            self.dir == (self.dir + 1) % 4
        elif action == Action.RIGHT:
            self.dir = (self.dir - 1 + 4) % 4



 
        
    def grow(self) -> None: 
        tail = self.body[self.length-2]  # -2 since head isn't inside body list
        self.body.append(pygame.Rect((tail.x, tail.y, self.block_size, self.block_size)))
        self.length+=1
        self.skip_tail_update = True

    # the singular inner tuple is the head while the list is the body
    def getBodyPositions(self) -> List[Tuple[int,int]]:
        return [utils.posToRowCol(block.topleft) for block in self.body]

    def getHeadPosition(self) -> Tuple[int, int]:
        return utils.posToRowCol(self.head.topleft)
     
    def draw(self, screen) -> None:
        pygame.draw.rect(screen, utils.RED, self.head)
        for block in self.body:
            pygame.draw.rect(screen, utils.BLUE, block)

    


        