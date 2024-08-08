from enum import Enum, IntEnum
from typing import Tuple
import numpy as np
import torch

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 50
WIDTH_IN_BLOCKS = SCREEN_WIDTH // GRID_SIZE
HEIGHT_IN_BLOCKS = SCREEN_HEIGHT// GRID_SIZE
MAX_LENGTH = WIDTH_IN_BLOCKS*HEIGHT_IN_BLOCKS
BOARD_SHAPE = (HEIGHT_IN_BLOCKS, WIDTH_IN_BLOCKS)

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

RED = (250, 0, 0)
BLUE = (0, 0, 250)
GREEN = (0, 250, 0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = 'cpu'
def getDirAsString(dir):
    if dir == 0:
        return "Up"
    elif dir == 1:
        return "down"
    elif dir == 2:
        return "left"
    elif dir == 3: 
        return "right"

class Direction(IntEnum):
    UP = 0,
    LEFT = 1,
    DOWN = 2,
    RIGHT = 3

class Action(IntEnum):
    NONE = 0,
    LEFT = 1,
    RIGHT = 2

class Objects(Enum):
    SNAKE_HEAD = 1,
    SNAKE_BODY = 2, 
    APPLE = 3,

ONE_HOT_DICT = {
    "empty": np.array([1,0,0,0]),
    "head": np.array([0,1,0,0]),
    "body": np.array([0,0,1,0]),
    "food": np.array([0,0,0,1])
}

OBJ_VAL_DICT = {
    "empty": 0,
    "head": 0.75,
    "body": 0.2,
    "food": 1
}

def posToRowCol(pixelPos: Tuple[int, int]) -> Tuple[int, int]:
    return [pixelPos[1] // GRID_SIZE, pixelPos[0] // GRID_SIZE]