import os
import time

import numpy as np
import cv2

from network import R2Plus1DClassifier
import torch
from glob import glob
from dataset import VideoDataset
from torch.utils.data import DataLoader
def recognize(fname, model,dataloader):
    # size = (60,30)
    # cap = cv2.VideoCapture(fname)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # buffer = np.empty((frame_count, size[1], size[0], 3), np.dtype('float32'))
    # count = 0
    
    # while count < frame_count:
    #     frame = cv2.resize(cap.read()[1], size)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     buffer[count] = frame
    #     count += 1
    # buffer = (buffer - np.mean(buffer)) / np.std(buffer)
    buffer = dataloader.loadvideo(fname)
    buffer = dataloader.normalize(buffer)
    buffer = torch.FloatTensor(buffer).permute(3,0,1,2).unsqueeze(0)

    outputs = model(buffer)
    _,preds = torch.max(outputs,1)
    commands = ['click_here', 'close_window', 'down_scroll', 'drag', 'drop_here', 'go_backward', 'go_forward',
                'scroll_up', 'search_this', 'zoom_in', 'zoom_out']
    # for s in sorted(list(zip(outputs.detach().numpy()[0],commands)),reverse=True):
    #     print(s)

    print(commands[preds[0].data])


if __name__ == "__main__":
    directory = "./zxsu/"
    val_set = VideoDataset(directory,mode = 'val')
    # restores the model and optimizer state_dicts
    lip_model = R2Plus1DClassifier(num_classes=11, layer_sizes=[3,3,3,3])
    state_dicts = torch.load("/home/rkmtlab/projects/zxsu/SilentCut_Oct/pure_model.pt", map_location = torch.device("cpu"))
    lip_model.load_state_dict(state_dicts)
    lip_model.eval()
    for f in glob("/home/rkmtlab/projects/zxsu/SilentCut_Oct/zxsu/click_here/*.avi"):
        recognize(f,lip_model,val_set)