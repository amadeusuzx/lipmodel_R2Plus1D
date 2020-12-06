import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import VideoDataset
from network import R2Plus1DClassifier

import sys
import cv2
import random
# Use GPU if available else revert to CPU

from torch_videovision.torchvideotransforms import video_transforms, volume_transforms

def train_model(num_classes, directory, path="model_data.pth.tar"):
    # batch_size = 20
    commands = sorted([
    'caption',
    'play',
    'stop',
    'go_back',
    'go_forward',
    'previous',
    'next',
    'volume_up',
    'volume_down',
    'maximize',
    'expand',
    'delete',
    'save',
    'like',
    'dislike',
    'share',
    'add_to_queue',
    'watch_later',
    'home',
    'trending',
    'subscription',
    'original',
    'library',
    'profile',
    'notification',
    'scroll_up',
    'scroll_down',
    'click'])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    folder = Path(directory)
    train_fnames,train_labels,val_fnames,val_labels = [],[],[],[]
    for label in sorted(os.listdir(folder)):
        shuffled_list = os.listdir(os.path.join(folder, label))
        random.Random(4).shuffle(shuffled_list)
        for fname in shuffled_list[:-10]:
            train_fnames.append(os.path.join(folder, label, fname))
            train_labels.append(label)
        for fname in shuffled_list[-10:]:
            val_fnames.append(os.path.join(folder, label, fname))
            val_labels.append(label)
    layer_sizes=[2,2,2,2,2,2]
    save=True
    # initalize the ResNet 18 version of this model
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)

    transforms = video_transforms.Compose([video_transforms.CenterCrop((30,60))])
    train_set = VideoDataset(fnames=train_fnames,labels=train_labels,transforms=transforms)
    train_set = VideoDataset(fnames=val_fnames,labels=val_labels,transforms=transforms)

    train_dataloader = DataLoader(train_set, batch_size = 1, shuffle=False, num_workers= 4)
    val_dataloader = DataLoader(train_set, batch_size = 1, shuffle=False, num_workers= 4)

    if os.path.exists(path):
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    dataloaders = {'train_dataloader':train_dataloader,'val_dataloader':val_dataloader}
    for phase in ['train_dataloader','val_dataloader']:
        i = 0  
        for inputs, labels in dataloaders[phase]:
            inputs_buffer = inputs.permute(0,4,1,2,3).to(device)

            with torch.set_grad_enabled(False):
                outputs = model.res2plus1d(inputs_buffer) 

            i += 1
            print(f"extracted {i} of {len(dataloaders[phase].dataset)} videos")
            feats_dir = f"features/{phase}/{commands[labels[0]]}"
            if not os.path.exists(feats_dir):
                os.makedirs(feats_dir) 
            np.save(f"{feats_dir}/{commands[labels[0]]}{i}.npy",outputs.cpu().detach().numpy())


if __name__ == "__main__":
    train_model(num_classes = int(sys.argv[1]),directory = sys.argv[2],path=sys.argv[3])
