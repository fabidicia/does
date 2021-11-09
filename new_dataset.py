################################# Various import #######################################

from __future__ import print_function, division
import cv2
import csv
import os.path
import os
from os import path
from csv import writer
from csv import reader
import argparse
import pathlib
from tqdm import tqdm
import os
import torch
import PIL
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from csv import reader
import random
import torchvision.transforms.functional as TF
import glob


################################ Dataset ########################

class HorizonDataset(Dataset):

    def __init__(self, base_folder, transform=None,filtered=None):

        self.transform = transform
        self.base_folder = base_folder
        self.txt_data_paths = glob.glob(base_folder + "/**/data.txt", recursive = True)
        self.subfolders = [os.path.dirname(path) for path in self.txt_data_paths]

        #Splitting the text data and lables from each other
        self.frame_list = []
        self.roll_list = []
        self.pitch_list = []

        #Opening the file and storing its contents in a list
        for txt_data_path, subfolder  in zip(self.txt_data_paths, self.subfolders):
            with open(txt_data_path) as data_file:

                self.rows = data_file.read().split('\n')   
                for row in (row for row in self.rows if row != ""):
                    self.frame_list.append(os.path.join(subfolder, row.split(',')[0].replace(":","")))
                    roll_idx = 6 if filtered else 3
                    pitch_idx = 7 if filtered else 4 
                    self.roll_list.append(row.split(',')[roll_idx]) 
                    self.pitch_list.append(row.split(',')[pitch_idx])

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, i):
            roll = torch.from_numpy(np.array(float(self.roll_list[i])))
            pitch = torch.from_numpy(np.array(float(self.pitch_list[i])))
            frame_file = self.frame_list[i]
            frame = PIL.Image.open(frame_file)



            if self.transform:
                frame = self.transform(frame)
            return roll, pitch, frame
