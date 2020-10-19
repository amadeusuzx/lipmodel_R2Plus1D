import os

import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

class VideoDataset(Dataset):

    def __init__(self, fnames, labels, mode='train', clip_len=8,transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.fnames, labels = fnames, labels
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        
    def __getitem__(self, index):

        buffer = self.loadvideo(self.fnames[index])
        if self.transforms:
            buffer = self.transforms(buffer)
        buffer = self.normalize(buffer)
        
        return buffer, self.label_array[index]
        
    def loadvideo(self, fname):
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        buffer = []

        count = 0
        retaining = True

        droplist = random.sample(range(frame_count),random.randint(0,frame_count//10)) if self.mode == "train" else []
        drop_beginning = random.randint(0,10)
        # sampling = np.linspace(0, frame_count-1, num=n_frame, dtype=int)
        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if (count > drop_beginning) and (count not in droplist):
                frame = cv2.resize(frame,(120,60))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = Image.fromarray(frame, 'RGB')
                buffer.append(frame)
            count += 1
        capture.release()
        return buffer 
               

    def normalize(self, buffer):
        # n_frame = 20
        new_buffer = []
        for b in buffer:
            new_buffer.append(np.array(b,"float32"))
        buffer = np.stack(new_buffer)
        buffer = (buffer - np.mean(buffer))/np.std(buffer)
        return buffer

    def __len__(self):
        return len(self.fnames)
