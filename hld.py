################################# Various import #######################################

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pickle
import io
import argparse
import math
from math import sin, cos, tan, pi, atan2, sqrt
import statistics
import pandas as pd 
import scipy.stats 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from random import randint
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,median_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from utils import ploton_tensorboard, ProgressParallel
from utils import normalize_roll, normalize_pitch, denormalize_roll, denormalize_pitch, inverse_normalize,LLSc
from new_dataset import HorizonDataset
from networks import Net
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
from joblib import Parallel, delayed

############################# Code call definition ######################################

parser = argparse.ArgumentParser("script to train horpe on oneplus videoframe")
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--method', type=str, default="hld1")
parser.add_argument('--train_path', type=str, default="./video_oneplus/Train_set")
parser.add_argument('--test_path', type=str, default="./video_oneplus/Test_set")
parser.add_argument('--roll', type=int, default=1)
parser.add_argument('--pitch', type=int, default=1)
parser.add_argument('--calibrated', type=int, default=1)
parser.add_argument('--bs_test', type=int, default=512)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--bright', type=float, default=0.0)
parser.add_argument('--contrast', type=float, default=0.0)
parser.add_argument('--saturation', type=float, default=0.0)
parser.add_argument('--hue', type=float, default=0.0)
parser.add_argument('--dropout', type=float)
parser.add_argument('--epochs', type=int,default=10)
parser.add_argument('--testtrain', action="store_true")
parser.add_argument('--test_hld', action="store_true")

args = parser.parse_args()
if args.method == "hld1": 
    from utils import hdl_predict
elif args.method == "hld3":
    from utils import hdl_predict3 as hdl_predict

############################# Image and data -related parameters definitions ######################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter('./runs/horpe_experiments')

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.bs
IMAGE_SIZE = 256
CROP_SIZE = 256
thr = 2.2
# the following values are obtained using the calc_mean_std.py script. UPDATE THEM basing on the dataset!!

IMG_MEAN=[0.5770, 0.6006, 0.6116] # NEW CORRECT VALUES, AS OPPOSED TO DEFAULT ONES: [0.485, 0.456, 0.406]
IMG_STD=[0.1698, 0.1686, 0.1778] # NEW CORRECT VALUES, AS OPPOSED TO DEFAULT ONES: [0.229, 0.224, 0.225]

# shift_roll = 90
# shift_pitch = 137
roll_mean = 0.650653185595568
roll_std = 6.653386182215193
pitch_mean = 93.7015324099723
pitch_std = 6.04184024823875


###train_dataset and test_dataset are build equal in this example. they will be made different by the next snippet of code
test_dataset = HorizonDataset(base_folder = args.test_path, transform=None)

def test_sample(dataset,i):
    roll, pitch, frame = dataset.__getitem__(i)
    pred_roll,pred_pitch = hdl_predict(frame)
    return roll, pred_roll, pitch, pred_pitch

def test_model(dataset):
    correct_roll = 0
    correct_pitch = 0
    total = 0
    roll_gt_list = []
    roll_pred_list = []
    roll_hld_pred_list = []
    pitch_gt_list = []
    pitch_pred_list = []
    from timeit import default_timer as timer
    start=timer()
    a,b,c,d=test_sample(dataset,0)
    end=timer()
    print(end - start)

    with torch.no_grad():
        results = ProgressParallel(n_jobs=8,total=len(dataset.frame_list))(delayed(test_sample)(dataset,i) for i in range(len(dataset.frame_list)))
        #results = [test_sample(dataset,i) for i in range(len(dataset.frame_list))]
        roll_gt_list = [elem[0] for elem in results]
        roll_pred_list = [elem[1] for elem in results]
        pitch_gt_list = [elem[2] for elem in results]
        pitch_pred_list = [elem[3] for elem in results]


    if args.roll:
        if args.calibrated:
            roll_pred_list = LLSc(roll_gt_list, roll_pred_list)
        residual_roll = (np.abs(np.asarray(roll_gt_list) - np.asarray(roll_pred_list)))
        correct_roll = np.sum(residual_roll <= thr)
        mae_roll = np.mean(residual_roll)
        std_roll = statistics.stdev(residual_roll.flatten())
        variance_roll = statistics.variance(residual_roll.flatten())
        med_roll = median_absolute_error(roll_gt_list,roll_pred_list)

        rmse_roll = sqrt(mean_squared_error(np.asarray(roll_gt_list), np.asarray(roll_pred_list)))

        print("Test: MAE hdl roll: %.4f" % mae_roll)
        print("Test: STD hdl roll: %.4f" % std_roll)
        print("Test: Variance hdl roll: %.4f" % variance_roll)
        print("Test: RMSE hdl roll: %.4f" % rmse_roll)
        print("Test: MED hdl roll: %.4f" % med_roll)
        print("Test: Roll accuracy w.r.t. arbitrary threshold: %d %%" % (100 * correct_roll / len(roll_gt_list)))

        fig, axs = plt.subplots(1, 2, tight_layout=True)        
        N, bins, patches = axs[0].hist(residual_roll, bins=20)
        axs[1].hist(residual_roll, bins=20, density=True)
        axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))


    if args.pitch:
        if args.calibrated:
            pitch_pred_list = LLSc(pitch_gt_list, pitch_pred_list)
        residual_pitch = (np.abs(np.asarray(pitch_gt_list) - np.asarray(pitch_pred_list)))
        correct_pitch = np.sum(residual_pitch <= thr)
        mae_pitch = np.mean(residual_pitch)
        std_pitch = statistics.stdev(residual_pitch.flatten())
        variance_pitch = statistics.variance(residual_pitch.flatten())
        med_pitch = median_absolute_error(pitch_gt_list,pitch_pred_list)

        rmse_pitch = sqrt(mean_squared_error(np.asarray(pitch_gt_list), np.asarray(pitch_pred_list)))

        print("Test: MAE hdl pitch: %.4f" % mae_pitch)
        print("Test: STD hdl pitch: %.4f" % std_pitch)
        print("Test: Variance hdl pitch: %.4f" % variance_pitch)
        print("Test: RMSE hdl pitch: %.4f" % rmse_pitch)
        print("Test: MED hdl pitch: %.4f" % med_pitch)
        print("Test: pitch accuracy w.r.t. arbitrary threshold: %d %%" % (100 * correct_pitch / len(pitch_gt_list)))

        fig, axs = plt.subplots(1, 2, tight_layout=True)        
        N, bins, patches = axs[0].hist(residual_pitch, bins=20)
        axs[1].hist(residual_pitch, bins=20, density=True)
        axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

test_model(test_dataset)


###least square model:
# gt=[gt1,gt2,gt3....] is the gt list, pred=[m1,m2,m3....] is the measurement list
# we wanna solve a system like that:
# gt1 = b + a*m1
# gt2 = b + a*m2
# gt3 = b + a*m3
#rewriting it as:
# a*m1 + b - gt1 = 0
# a*m2 + b - gt2 = 0
# a*m3 + b - gt3 = 0
#in matrix form:
#                x=[a]
#                  [b]
#
#                A=[m1 1]
#                  [m2,1]
#                  [m3,1]

#                b=[gt1]
#                  [gt2]
#                  [gt3]

#so that the final form is:
#  Ax - b = 0
#such system is sovradetermined, so we can minimize the squared norm of |Ax-b|
#in python code:
#with num equal to the number of measurements to fit,we can write:
#A = np.zeros((num,2),dtype=np.float)
#A[:,0] = pred_list
#A[:,1] = 1.0

#b = np.zeros((num,),dtype=np.float)
#b[:] = gt_list
#res = lsq_linear(A, b, verbose=0)
#solution x=[a,b] is in res.x
#so now to calibrate the system minimizing the squared sum of errors we just have to multiply pred_list by a and sum b.

#for hld3 res.x= array([ 4.06594286e-03, -9.62077696e+01])
#so:
#pred_list = np.array(pred_list)*res.x[0]+res.x[1]
