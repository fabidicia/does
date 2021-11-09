import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pickle
import io
import PIL 
from PIL import Image
from torchvision.transforms import ToTensor
from random import randint
import argparse
from math import sin, cos, tan, pi, atan2, sqrt
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
#from utils import plot_tensorboard
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
torch.backends.cudnn.benchmark = True
from new_dataset import HorizonDataset
from tqdm import tqdm

#################################### functions ##########################à

def plot_tensorboard(writer,Datas, Lines, Labels,Name="Image",ylabel=None,ylim=None):
    rnd = randint(0,10000)
    with open(Name+".pkl", "wb") as open_file:
        pickle.dump(Datas, open_file)

    if Name == "Image":
        Name = Name + str(rnd)
    plt.figure(rnd,figsize=(14,7))
        ## code to plot the image in tensorboard
    plt.title(Name)
    if ylabel is not None:
         plt.ylabel(ylabel)
    plt.xlabel("Time [0.01s]")
    if ylim is not None:
        axes = plt.gca()
        axes.set_ylim(ylim)
    times = [i for i in range(len(Datas[0]))]
    for data,line,label in zip(Datas, Lines, Labels):
        plt.plot(times, data, line,label=label)
        plt.legend(loc="lower right")

    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    writer.add_image(Name, image, 0)
    plt.clf()
    plt.cla()
    plt.close()
    

def normalize_roll(roll):
    return (roll - roll_mean) / roll_std  #substract the mean, divide by std, so that now it has mean 0 std 1

def normalize_pitch(pitch):
    return (pitch - pitch_mean) / pitch_std 

def denormalize_roll(roll):
    return (roll * roll_std) + roll_mean   #multiply by std, add the mean

def denormalize_pitch(pitch):
    return (pitch * pitch_std) + pitch_mean   #multiply by std, add the mean

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor



#########################à Dataset elaboration #################################

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--train_path', type=str, default="./video_oneplus/Train_set")
args = parser.parse_args()

#IMG_MEAN = [0.5770, 0.6008, 0.6116]
#IMG_STD = [0.1698, 0.1686, 0.1778]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
#    transforms.Normalize(IMG_MEAN, IMG_STD),
])

train_dataset = HorizonDataset(base_folder = args.train_path,
                                transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=128,
                                                   shuffle=False,
                                                   num_workers=8) 

def online_mean_and_sd(loader):

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for _,_,images in tqdm(loader):

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# mean, std = online_mean_and_sd(train_loader)
# print("img mean: ")
# print(mean)
# print("img std: ")
# print(std)

    ############################# normalization and denormalization functions #################

class Normalization:
    def __init__(self, loader):
        tmp_roll = []
        tmp_pitch = []
        for roll, pitch, _ in loader:
            for item in roll:
                tmp_roll.append(item.numpy())
            for item in pitch:
                tmp_pitch.append(item.numpy())
        tmp_roll = np.asarray(tmp_roll)
        tmp_pitch = np.asarray(tmp_pitch)
        self.roll_mean = tmp_roll.mean()
        self.roll_std = tmp_roll.std()
        self.pitch_mean = tmp_pitch.mean()
        self.pitch_std = tmp_pitch.std()
        self.shift_roll= abs(round(min(tmp_roll)))+1
        self.shift_pitch = abs(round(min(tmp_pitch)))+1
        
        self.min_roll = round(min(tmp_roll))
        self.max_roll = round(max(tmp_roll))
        self.min_pitch = round(min(tmp_pitch))
        self.max_pitch = round(max(tmp_pitch))

    def normalize_roll(self,roll):
        return (roll - self.roll_mean) / self.roll_std  #substract the mean, divide by std, so that now it has mean 0 std 1
 
    def normalize_pitch(self,pitch):
        return (pitch - self.pitch_mean) / self.pitch_std 

    def denormalize_roll(self,roll):
        return (roll * self.roll_std) + self.roll_mean   #multiply by std, add the mean

    def denormalize_pitch(self,pitch):
        return (pitch * self.pitch_std) + self.pitch_mean   #multiply by std, add the mean


norm = Normalization(train_loader)

shift_roll = norm.shift_roll
shift_pitch = norm.shift_pitch
roll_mean = norm.roll_mean
roll_std = norm.roll_std
pitch_mean = norm.pitch_mean
pitch_std = norm.pitch_std

total_range_roll= (abs(norm.min_roll)) + norm.max_roll + 1
total_range_pitch = (abs(norm.min_pitch)) + norm.max_pitch + 1


print('roll_mean is: ' + str(roll_mean) + '; pitch_mean is: ' + str(pitch_mean) + '; roll_std is: ' + str(roll_std) + '; pitch_std is: ' + str(pitch_std))
print('the shift for roll is: ' + str(shift_roll) + '; the shift for pitch is: ' + str(shift_pitch))
print('the total range of roll values is: ' + str(total_range_roll) + '; the total range of pitch is: ' + str(total_range_pitch))
