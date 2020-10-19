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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from torch_videovision.torchvideotransforms import video_transforms, volume_transforms

def pad_3d_sequence(batch):
    batch_ = list(zip(*batch))
    sequences = [torch.Tensor(b) for b in batch_[0]]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.LongTensor(batch_[1]).squeeze()
    return sequences_padded, labels

def train_model(num_classes, directory, num_epochs=45, path="model_data.pth.tar"):
    # batch_size = 20
    folder = Path(directory)
    train_fnames,train_labels,val_fnames,val_labels = [],[],[],[]
    for label in sorted(os.listdir(folder)):
        shuffled_list = os.listdir(os.path.join(folder, label))
        random.shuffle(shuffled_list)
        for fname in shuffled_list[:]:
            train_fnames.append(os.path.join(folder, label, fname))
            train_labels.append(label)
        for fname in shuffled_list[-5:]:
            val_fnames.append(os.path.join(folder, label, fname))
            val_labels.append(label)  
    layer_sizes=[2,2,2,2,2]
    save=True
    # initalize the ResNet 18 version of this model
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)
    model = torch.nn.DataParallel(model, device_ids=[1,2,0])

    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in pa per sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs


    video_transform_list = [
        video_transforms.RandomRotation((5)),
        video_transforms.RandomResize((1,1.2)),
        video_transforms.CenterCrop((48,86)),    # h,w
        video_transforms.ColorJitter(0.3,0.3,0.3)]
    transforms = video_transforms.Compose(video_transform_list)

    test_transforms = video_transforms.Compose([video_transforms.CenterCrop((48,86))])
    train_set = VideoDataset(fnames=train_fnames,labels=train_labels,transforms=transforms)
    val_set = VideoDataset(fnames=val_fnames,labels=val_labels,mode = 'val',transforms=test_transforms)

    train_dataloader = DataLoader(train_set, batch_size = 12, shuffle=True, num_workers=8 , collate_fn = pad_3d_sequence)

    val_dataloader = DataLoader(val_set, batch_size=1, num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}


    start = time.time()
    epoch_resume = 0

    if os.path.exists(path):
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':

                scheduler.step()
                model.train()
            else:
                model.eval()
            
            for inputs, labels in dataloaders[phase]:
                inputs_buffer = inputs.permute(0,4,1,2,3).to(device)
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
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': epoch_acc,
            'opt_dict': optimizer.state_dict(),
            }, path)
            model.eval()
            torch.save(model.module.state_dict(), path+"_puremodel")

    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")


if __name__ == "__main__":
    train_model(num_classes = int(sys.argv[1]),directory = sys.argv[2],num_epochs=int(sys.argv[3]),path=sys.argv[4])
