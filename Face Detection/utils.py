import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 224

NUM_POINTS = 68

batch_size = 64