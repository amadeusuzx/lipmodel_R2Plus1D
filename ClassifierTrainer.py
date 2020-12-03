import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from pathlib import Path
from network import R2Plus1DClassifier

import sys
import cv2
import random
# Use GPU if available else revert to CPU

from torch_videovision.torchvideotransforms import video_transforms, volume_transforms
class ClassifierDataset(Dataset):

    def __init__(self, fnames, labels):

        self.fnames, labels = fnames, labels
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        
    def __getitem__(self, index):

        buffer = np.load(self.fnames[index])[0]
        return buffer, self.label_array[index]

    def __len__(self):
        return len(self.fnames)
def train_model(num_classes, directory,num_epochs,model_name):
    # batch_size = 20
    commands = sorted([
    'caption',
    'play',
    'stop',
    'go_back',
    'go_forward',
    'previous',
    'next'])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    folder = Path(directory)
    train_fnames,train_labels,val_fnames,val_labels = [],[],[],[]
    for label in sorted(os.listdir(folder)):
        if label in commands:
            shuffled_list = os.listdir(os.path.join(folder, label))
            random.shuffle(shuffled_list)
            for fname in shuffled_list[:-3]:
                train_fnames.append(os.path.join(folder, label, fname))
                train_labels.append(label)
            for fname in shuffled_list[-3:]:
                val_fnames.append(os.path.join(folder, label, fname))
                val_labels.append(label)
    layer_sizes=[2,2,2,2,2,2]
    save=True
    # initalize the ResNet 18 version of this model
    model = nn.Linear(1024, num_classes).to(device)

    train_set = ClassifierDataset(fnames=train_fnames,labels=train_labels)
    train_dataloader = DataLoader(train_set, batch_size = 16, shuffle=True, num_workers= 1)
    val_set = ClassifierDataset(fnames=val_fnames,labels=val_labels)
    val_dataloader = DataLoader(val_set, batch_size = 1, shuffle=True, num_workers= 1)
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in pa per sec 4.1

    train_dataloader = DataLoader(train_set, batch_size = 3, shuffle=True, num_workers= 16)

    val_dataloader = DataLoader(val_set, batch_size=1, num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    start = time.time()
    epoch_resume = 0

    for epoch in tqdm(range(0, num_epochs), unit="epochs", initial=0, total=num_epochs):
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            for inputs, labels in dataloaders[phase]:
                inputs_buffer = inputs.to(device)
                labels_buffer = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs_buffer)

                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels_buffer)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs_buffer.size(0)
                running_corrects += torch.sum(preds == labels_buffer.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"\n{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

        if save:
            model.eval()
            torch.save(model.state_dict(), model_name)

    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")

if __name__ == "__main__":
    train_model(num_classes = int(sys.argv[1]),directory = sys.argv[2],num_epochs = int(sys.argv[3]),model_name =sys.argv[4] )
