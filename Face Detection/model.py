import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from utils import IMG_SIZE, DEVICE, NUM_POINTS



def calculate_conv_out_dim(kernel_sizes_list, pool_sizes_list, input_size = IMG_SIZE):
    conv_out_dim = input_size
    for kernel_size, pool_size in zip(kernel_sizes_list, pool_sizes_list):
        conv_out_dim = (conv_out_dim - kernel_size + 1) // pool_size
    return conv_out_dim
    
    
    
    
def conv2d_model(conv_out_1: int = 32, conv_k_1 : int = 5,
                 conv_out_2: int = 64, conv_k_2 : int = 3, 
                 conv_out_3: int = 128, conv_k_3 : int = 3, 
                 N : int = 2, p : float = 0.2):
    conv_out_dim = calculate_conv_out_dim([conv_k_1, conv_k_2, conv_k_3], [N, N, N], IMG_SIZE)
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=conv_out_1, 
                  kernel_size=(conv_k_1, conv_k_1)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=N, stride=N),
        
        nn.Conv2d(in_channels=conv_out_1, out_channels=conv_out_2, 
                  kernel_size=(conv_k_2, conv_k_2)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=N, stride=N),
        
        nn.Conv2d(in_channels=conv_out_2, out_channels=conv_out_3, 
                  kernel_size=(conv_k_3, conv_k_3)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=N, stride=N),
        
        nn.Dropout(p),
        
        nn.Flatten(),   
        nn.Linear(conv_out_3 * (conv_out_dim**2), NUM_POINTS*2) #*2 since need x,y for each pt
    )
    model.to(DEVICE)
    return model



def train(model: nn.Module, optimizer: optim, 
          train_loader: DataLoader, val_loader: DataLoader, 
          epochs: int = 50
          ) -> Tuple[List[float], List[float], List[float], List[float]]:
    
    loss = nn.MSELoss()
    train_losses = []
    train_avg_distances = []
    val_losses = []
    val_avg_distances = []
    train_len = len(train_loader)
    val_len = len(val_loader)
    
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_avg_dist = 0.0
        
        for(img, keypts) in train_loader:
            optimizer.zero_grad()
            keypts_pred = model(img)
            keypts_pred = keypts_pred.reshape(-1, NUM_POINTS, 2)
            batch_loss = loss(keypts_pred, keypts)
            train_loss += batch_loss.item()
            
            dists = torch.sqrt(torch.sum((keypts_pred - keypts)**2, dim = 2))
            batch_avg_dist = dists.mean().item()
            train_avg_dist += batch_avg_dist
            
            batch_loss.backward()
            optimizer.step()
        
        train_losses.append(train_loss/train_len)
        train_avg_distances.append(train_avg_dist/train_len)
        
        
        
        model.eval()
        val_loss = 0.0
        val_avg_dist = 0.0
        with torch.no_grad():
            for (img, keypts) in val_loader:
                optimizer.zero_grad()
                keypts_pred = model(img)
                keypts_pred = keypts_pred.reshape(-1, NUM_POINTS, 2)
                batch_loss = loss(keypts_pred, keypts)
                val_loss += batch_loss.item()
                
                dists = torch.sqrt(torch.sum((keypts_pred - keypts)**2, dim = 2))
                batch_avg_dist = dists.mean().item()
                val_avg_dist += batch_avg_dist
            val_losses.append(val_loss/val_len)
            val_avg_distances.append(val_avg_dist/val_len)
    
    return train_losses, train_avg_distances, val_losses, val_avg_distances

def param_search(train_loader: DataLoader, val_loader: DataLoader, num_combos) -> Dict[str, Any]:
    # maps best_val_loss to an inner dictionarry
    # that contains lr, dim, model, and another inner dictionary
    # with loss and accuracy for train and val
    data_dict = {}

    # for i in range(iterations):
    # M_choices = [70, 80, 90, 100, 110, 120, 130]
    M_choices = [32]

    best_avg_dist = float('inf')
    for M in M_choices:
        for i in range(num_combos):
            lr = np.random.uniform(1e-3, 1e-2)
            k = np.random.randint(3, 7)
            N = np.random.randint(2, 10)
            model = conv2d_model()
            print()
            print("M = ", M, " | lr = ", lr, " | k = ", k, " | N = ", N)

            optim = torch.optim.Adam(model.parameters(), lr)
            train_loss, train_avg_dists, val_loss, val_avg_dists = train(
                model,
                optim,
                train_loader,
                val_loader,
                epochs=100
            )
            
            print(min(val_avg_dists))
            if min(val_avg_dists) < best_avg_dist:
                best_avg_dist = min(val_avg_dists)
                print("new best = ", best_avg_dist)
                data_dict["params"] =  {
                        "lr": lr,
                        "M" : M,
                        "k" : k,
                        "N" : N,
                    }
                data_dict["plot_data"] = {
                        "train_losses": train_loss,
                        "train_avg_dists": train_avg_dists,
                        "val_losses": val_loss,
                        "val_avg_dists" : val_avg_dists,
                    }
                data_dict["model"] = model
    return data_dict
            
