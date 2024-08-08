import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from model import param_search, conv2d_model
from dataloader_300w import load_300w_dataset

def main():
    print(utils.DEVICE)
    # indoor_dataset, outdoor_dataset = dataLoader.load_300w_data()

    # indoor_train, indoor_val, indoor_test = dataLoader.split_dataset(indoor_dataset, 0.15, 0.15)
    # outdoor_train, outdoor_val, outdoor_test = dataLoader.split_dataset(outdoor_dataset, 0.15, 0.15)

    # train_dataset = ConcatDataset([indoor_train, outdoor_train])
    # val_dataset = ConcatDataset([indoor_val, outdoor_val])
    
    # train_loader = DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=utils.batch_size, shuffle=True)
    # indoor_test_loader = DataLoader(indoor_test, batch_size=utils.batch_size, shuffle=True)

    train_dataset = load_300w_dataset("./Data/lfpw/trainset", "./Data/300W Boxes/bounding_boxes_lfpw_trainset.mat", True)
    train_dataset, val_dataset = random_split(train_dataset, [0.85, 0.15])
    test_dataset = load_300w_dataset("./Data/lfpw/testset",  "./Data/300W Boxes/bounding_boxes_lfpw_testset.mat", True)
    
    train_loader = DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=utils.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=utils.batch_size, shuffle=True)

    
    res = param_search(train_loader, val_loader, 1)
    model = res["model"]
    torch.save(model.state_dict(), "model_state.pth")

    for param in res["params"].keys():
        print(param + " = ", res["params"][param])
    
    
    # data = res["plot_data"]
    # plt.figure()
    # plt.plot(res["plot_data"]["train_avg_dists"], label ="train avg dist")
    # plt.plot(data["val_avg_dists"], label = "val avg dist")
    # plt.title("Average Distances")
    # plt.xlabel("Epoch")
    # plt.ylabel("Avg Distance")
    # plt.legend()
    
    # plt.figure()
    # plt.plot(data["train_losses"], label ="train loss")
    # plt.plot(data["val_losses"], label = "val loss")
    # plt.title("Losses")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    
    # plt.show()
    
    
    for img_batch, pts_batch in test_loader:
        for img, pts in zip(img_batch, pts_batch):
            img_cpu = img.cpu()
            pts_cpu = pts.cpu()
            image_exp = img_cpu.numpy()
            image_exp = np.transpose(image_exp, (1, 2, 0))
            if image_exp.max() <= 1.0:
                image_exp = (image_exp * 255).astype(np.uint8)
                
            
            image_exp = cv2.cvtColor(image_exp, cv2.COLOR_RGB2BGR)
            pts_cpu *= image_exp.shape[0]
            for x,y in pts_cpu:
                cv2.circle(image_exp, (int(x), int(y)), 1, (0,255,0), -1)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(image_exp, cv2.COLOR_BGR2RGB))  
            axes[0].axis('off') 
            axes[0].set_title("expected")
            
            
            with torch.no_grad():
                if img.dim() == 3:  # (channels, height, width)
                    img = img.unsqueeze(0)
                pred_pts = model(img)
                pred_pts = pred_pts.reshape(utils.NUM_POINTS, 2)
                
                
                image_pred = img.cpu().numpy().squeeze(0) 
                image_pred = np.transpose(image_pred, (1, 2, 0))
                if image_pred.max() <= 1.0:
                    image_pred = (image_pred * 255).astype(np.uint8)
                    
                image_pred = cv2.cvtColor(image_pred, cv2.COLOR_RGB2BGR)
                
                pred_pts *= image_pred.shape[0]
                for x,y in pred_pts:
                    cv2.circle(image_pred, (int(x), int(y)), 1, (0,255,0), -1)
                axes[1].imshow(cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB))  
                axes[1].axis('off') 
                axes[1].set_title("predicted")

            plt.show()
    
if __name__ == "__main__":
    main()