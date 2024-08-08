import os
import cv2
import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np
from scipy.io import loadmat
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Any

from utils import DEVICE, IMG_SIZE, NUM_POINTS


MAT_IMG_NAME_IDX = 0
MAT_BOX_IDX = 2





def load_300w_dataset(imgs_path: str, boxes_path: str, load_from_save : bool = True):
    path = imgs_path+".pth"
    if os.path.isfile(path) and load_from_save:
        data_dict = torch.load(path)
        dataset = TensorDataset(data_dict["images"], data_dict["keypoints"])
    else:
        dataset = load_imgs_and_pts(imgs_path, boxes_path)
        imgs, keypts = dataset.tensors
        torch.save({"images": imgs, "keypoints": keypts}, path)
    return dataset
        
    
def load_imgs_and_pts(dir: str, boxes_path: str):
    try:
        files = os.listdir(dir)
        boxes = loadmat(boxes_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    
    n = len(files)
    images_dataset = torch.zeros((n//2, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=DEVICE)
    points_dataset = torch.zeros((n//2, NUM_POINTS, 2), dtype=torch.float32, device=DEVICE)
    
    boxes = boxes["bounding_boxes"][0]
    if(len(boxes) != n//2):
        print(len(boxes), n)
        raise RuntimeError("Boxes mat file doesn't match with img directory")
    
    
    for i in tqdm(range(0, n-1, 2)):
        img_name = files[i]
        box_info = flatten_matlab_struct(boxes[i//2])
        
        if(box_info[MAT_IMG_NAME_IDX] != img_name):
            raise RuntimeError("Boxes mat file doesn't match with img directory")
    
        img_path = os.path.join(dir, files[i])
        pts_path = os.path.join(dir, files[i+1])
        
        if not os.path.isfile(img_path) or not os.path.isfile(pts_path):
            raise FileNotFoundError

        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = load_points(pts_path)

        cropped_image, adjusted_points = crop_image_and_adjust_pts(image, flatten_matlab_struct(box_info[2]), points, 10)
        
        # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
        # axes[0].axis('off') 
        # axes[0].set_title("original")
        
        # axes[1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  
        # axes[1].axis('off') 
        # axes[1].set_title("original")
        
        # plt.show()
        
        
        image_tensor, points_tensor, _, _, _ = scale_img_and_pts(cropped_image, adjusted_points, (IMG_SIZE,IMG_SIZE))
        images_dataset[i//2] = image_tensor
        points_dataset[i//2] = points_tensor
    
    return TensorDataset(images_dataset, points_dataset)


def flatten_matlab_struct(mat_struct):
    # Unwrap single-element lists and arrays
    while isinstance(mat_struct, (list, tuple, np.ndarray)) and len(mat_struct) == 1:
        mat_struct = mat_struct[0]
    return mat_struct

def crop_image_and_adjust_pts(image, bounds, points, margin):
    xmin, ymin, xmax, ymax = bounds
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(image.shape[1], xmax + margin)
    ymax = min(image.shape[0], ymax + margin)
    
    adjusted_points = points - torch.tensor([xmin, ymin], dtype = torch.float32, device=points.device)
    return image[ymin:ymax, xmin:xmax], adjusted_points



def load_points(path: str):
    res = torch.zeros((NUM_POINTS, 2), dtype=torch.float32, device=DEVICE)
    with open(path, 'r') as file:
       
        #skip the version, number of points, and opening brace
        for i in range(0,3):
            next(file)
            
        for i in range(0, NUM_POINTS):
            line = next(file)
            tokens = line.split()
            point = torch.tensor([float(tokens[0]), float(tokens[1])])
            res[i] = point
            
    return res
    
    



def scale_img_and_pts(image, points, target_size):
    h, w, _ = image.shape
    target_h, target_w = target_size
    
    # Calculate scale and new size
    scale = min(target_h / h, target_w / w)
    new_w = int(scale * w)
    new_h = int(scale * h)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a padded image
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
        
    to_tensor = torchvision.transforms.ToTensor()
    padded_image_tensor = to_tensor(padded_image)
    padded_image_tensor.to(DEVICE)
    
    padding = torch.tensor([left, top], dtype=torch.float32, device=points.device)
    scaled_points = points * scale
    scaled_points=torch.add(scaled_points, padding)
    scaled_points[:, 0] /= target_w
    scaled_points[:, 1] /= target_h

    return padded_image_tensor, scaled_points, scale, top, left




def split_dataset(dataset: TensorDataset, test_frac:float, val_frac:float):
    total_size = len(dataset)
    test_size = int(total_size*test_frac)
    val_size = int(total_size*val_frac)
    train_size = total_size - test_size - val_size

    if(train_size <= 0):
        raise ValueError("test and validation set fractions are too large")
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset