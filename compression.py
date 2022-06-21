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
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sns.set_theme()
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
from utils import ploton_tensorboard, SquarePad, Onnx_Model
from utils import normalize_roll, normalize_pitch, denormalize_roll, denormalize_pitch, inverse_normalize,remove_parameters
from new_dataset import HorizonDataset
from networks import Net
torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import ImageFile
from torch.nn.utils import prune
from flopco import FlopCo
from musco.pytorch import CompressorVBMF





ImageFile.LOAD_TRUNCATED_IMAGES = True
############################# Code call definition ######################################

parser = argparse.ArgumentParser("script to train horpe on oneplus videoframe")
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--train_path', type=str, default="./ROPIS_dataset/Train_set/Train_set_beach")
parser.add_argument('--test_path', type=str, default="./ROPIS_dataset/Test_set/Test_set_beach")
parser.add_argument('--roll', type=int, default=1)
parser.add_argument('--pitch', type=int, default=1)
parser.add_argument('--bs_test', type=int, default=512)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--bright', type=float, default=0.0)
parser.add_argument('--contrast', type=float, default=0.0)
parser.add_argument('--saturation', type=float, default=0.0)
parser.add_argument('--hue', type=float, default=0.0)
parser.add_argument('--dropout', type=float)
parser.add_argument('--epochs', type=int,default=10)
parser.add_argument('--testtrain', action="store_true")
parser.add_argument('--test_hld', action="store_true")
parser.add_argument('--filtered', default=1)
parser.add_argument('--pruning', type=float, default=0)
parser.add_argument('--compression', action="store_true")
parser.add_argument('--quantization', action="store_true")

args = parser.parse_args()

############################# Image and data -related parameters definitions ######################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_cpu = torch.device('cpu')
seed = randint(0,1000)
exper_path = "./runs/"+str(seed)+"/"
print("experiment path: "+exper_path)
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)

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

##################### make some augmentations on training data ####################
train_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((256,256)),
#    transforms.Resize((512, 288)),
#    transforms.CenterCrop((CROP_SIZE,CROP_SIZE)),
    transforms.ColorJitter(brightness=args.bright, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])
test_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((256,256)),
#    transforms.Resize((512, 288)),
#    transforms.CenterCrop((CROP_SIZE,CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)])

denormalize = transforms.Normalize(
   mean=[-IMG_MEAN[0]/IMG_STD[0], -IMG_MEAN[1]/IMG_STD[1], -IMG_MEAN[2]/IMG_STD[2]],
   std=[1/IMG_STD[0], 1/IMG_STD[1], 1/IMG_STD[2]]
)

###train_dataset and test_dataset are build equal in this example. they will be made different by the next snippet of code

train_dataset = HorizonDataset(base_folder = args.train_path, transform=train_transform,filtered=args.filtered)
aa = train_dataset[1000]
test_dataset = HorizonDataset(base_folder = args.test_path, transform=test_transform,filtered=args.filtered)


############################# Ptorch data loader ####################################

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) 

test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test, shuffle=False, num_workers=8)
testtrain_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_test, shuffle=False, num_workers=8)

############################# Model definition ######################################

model = Net(args)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr_sch = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion_mse = torch.nn.MSELoss() 

############################# Train and Test Model definition ######################################

def test_model(model,dataloader,curr_epoch=0):
    model.eval()
    correct_roll = 0
    correct_pitch = 0
    total = 0
    roll_gt_list = []
    roll_pred_list = []
    roll_hld_pred_list = []
    pitch_gt_list = []
    pitch_pred_list = []

    with torch.no_grad():
        for roll, pitch, frame in dataloader: ##### è dataloader o data_loader?
            frame = frame.to(device, dtype=torch.float)
            total += frame.size(0) #questo è il batch size

            if args.roll:
                roll = roll.to(device, dtype=torch.float)
                roll = normalize_roll(roll, roll_mean, roll_std)
                roll = roll.view(roll.size(0), -1) 
                out_reg_roll, _ = model(frame)
                roll_gt_list += (denormalize_roll(roll, roll_mean, roll_std)).tolist() 
                roll_pred_list += (denormalize_roll(out_reg_roll, roll_mean, roll_std)).tolist()    ##aggiungo una dimensione per matchare la shape di outputs!

            if args.pitch:
                pitch = pitch.to(device, dtype=torch.float)
                pitch = normalize_pitch(pitch, pitch_mean, pitch_std)
                pitch = pitch.view(pitch.size(0), -1) 
                _, out_reg_pitch = model(frame)
                pitch_gt_list += (denormalize_pitch(pitch, pitch_mean, pitch_std)).tolist() 
                pitch_pred_list += (denormalize_pitch(out_reg_pitch, pitch_mean, pitch_std)).tolist()

    if args.roll:
        residual_roll = (np.abs(np.asarray(roll_gt_list) - np.asarray(roll_pred_list)))
        correct_roll = np.sum(residual_roll <= thr)
        mae_roll = np.mean(residual_roll)
        std_roll = statistics.stdev(residual_roll.flatten())
        variance_roll = statistics.variance(residual_roll.flatten())
        mare_roll = np.mean(abs((residual_roll)/np.asarray(roll_gt_list)))
        rmse_roll = sqrt(mean_squared_error(np.asarray(roll_gt_list), np.asarray(roll_pred_list)))
        mape_roll = mean_absolute_percentage_error(roll_gt_list,roll_pred_list)
        med_roll = median_absolute_error(roll_gt_list,roll_pred_list)
        print("Test: MAE gt-horpe_roll: %.4f" % mae_roll)
        print("Test: STD gt-horpe_roll: %.4f" % std_roll)
        print("Test: Variance gt-horpe_roll: %.4f" % variance_roll)
        print("Test: MARE gt-horpe_roll: %.4f" % mare_roll)
        print("Test: MAPE gt-horpe_roll: %.4f" % mape_roll)
        print("Test: MED gt-horpe_roll: %.4f" % med_roll)
        print("Test: RMSE gt-horpe_roll: %.4f" % rmse_roll)
        print("Test: Roll accuracy w.r.t. arbitrary threshold: %d %%" % (100 * correct_roll / len(roll_gt_list)))
        writer.add_scalar('Loss/Roll/MAE', mae_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/STD', std_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/VAR', variance_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/MARE', mare_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/MAPE', mape_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/MED', med_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/RMSE', rmse_roll, curr_epoch)
        writer.add_scalar('Loss/Roll/Acc', (100 * correct_roll / len(roll_gt_list)), curr_epoch)

        ### TENSORBOARD PLOTTING
        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel(" Roll Absolute Error between predicted and GT values [deg]")
        bins = [elem for elem in range(0,11,1)]
        values, base, _ = plt.hist( residual_roll  , bins = bins, density=True, color = "darkgreen", range = None, label = "Histogram",histtype='stepfilled')
        writer.add_figure(tag="roll_AE",figure=fig,global_step=curr_epoch)
        writer.flush()

        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel(" Roll Cumulative Absolute Error between predicted and GT values [deg]")
        bins = [elem for elem in range(0,11,1)]
        values, base, _ = plt.hist( residual_roll  , bins = bins, density=True,cumulative=True, color = "limegreen", range = None, label = "Histogram",histtype='stepfilled')
        writer.add_figure(tag="roll_Cumulative",figure=fig,global_step=curr_epoch)
        writer.flush()

        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel("Roll Empirical Cumulative Distribution of Error [deg]")
        bins = [elem for elem in range(0,11,1)]
        sns.ecdfplot(data=residual_roll,legend=False,palette="BuGn")
        writer.add_figure(tag="roll_ECDF",figure=fig,global_step=curr_epoch)
        writer.flush()

    if args.pitch:
        residual_pitch = np.abs(np.asarray(pitch_gt_list) - np.asarray(pitch_pred_list))
        std_pitch = statistics.stdev(residual_pitch.flatten())
        correct_pitch = np.sum(residual_pitch <= thr)
        variance_pitch = statistics.variance(residual_pitch.flatten())
        mae_pitch = np.mean(residual_pitch)
        mare_pitch = np.mean(abs((residual_pitch)/np.asarray(pitch_gt_list)))
        rmse_pitch = sqrt(mean_squared_error(np.asarray(pitch_gt_list), np.asarray(pitch_pred_list)))
        mape_pitch = mean_absolute_percentage_error(pitch_gt_list,pitch_pred_list)
        med_pitch = median_absolute_error(pitch_gt_list,pitch_pred_list)

        print("Test: MAE gt-horpe_pitch: %.4f" % mae_pitch)
        print("Test: STD gt-horpe_pitch: %.4f" % std_pitch)
        print("Test: Variance gt-horpe_pitch: %.4f" % variance_pitch)
        print("Test: MARE gt-horpe_pitch: %.4f" % mare_pitch)
        print("Test: MAPE gt-horpe_pitch: %.4f" % mape_pitch)
        print("Test: MED gt-horpe_pitch: %.4f" % med_pitch)
        print("Test: RMSE gt-horpe_pitch: %.4f" % rmse_pitch)
        print("Test: Pitch accuracy w.r.t. arbitrary threshold: %d %%" % (100 * correct_pitch / len(pitch_gt_list)))
        writer.add_scalar('Loss/Pitch/MAE', mae_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/STD', std_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/VAR', variance_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/MARE', mare_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/MAPE', mape_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/MED', med_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/RMSE', rmse_pitch, curr_epoch)
        writer.add_scalar('Loss/Pitch/Acc', (100 * correct_pitch / len(pitch_gt_list)), curr_epoch)

        ### TENSORBOARD PLOTTING
        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel(" Pitch Absolute Error between predicted and GT values [deg]")
        bins = [elem for elem in range(0,11,1)]
        values, base, _ = plt.hist( residual_pitch  , bins = bins, density=True, color = "navy", range = None, label = "Histogram",histtype='stepfilled')
        writer.add_figure(tag="pitch_AE",figure=fig,global_step=curr_epoch)
        writer.flush()

        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel(" Pitch Cumulative Absolute Error between predicted and GT values [deg]")
        bins = [elem for elem in range(0,11,1)]
        values, base, _ = plt.hist( residual_pitch  , bins = bins, density=True,cumulative=True, color = "cornflowerblue", range = None, label = "Histogram",histtype='stepfilled')
        writer.add_figure(tag="pitch_Cumulative",figure=fig,global_step=curr_epoch)
        writer.flush()

        fig = plt.figure()
        ax = plt.axes()
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
        plt.ylabel("Total percentage of samples")
        plt.xlabel("Pitch Empirical Cumulative Distribution of Error [deg]")
        bins = [elem for elem in range(0,11,1)]
        sns.ecdfplot(data=residual_pitch,legend=False,palette="Blues")
        writer.add_figure(tag="pitch_ECDF",figure=fig,global_step=curr_epoch)
        writer.flush()


def train_model(model, data_loader, dataset_size, optimizer, scheduler, num_epochs):
    day = datetime.today().strftime("%b-%d-%Y")
    weights_filename = args.model + "_" + day

    print("Run the command 'tensorboard --logdir=runs' and navigate to the given link to see the images")
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        #scheduler.step() DISABLED
        model.train()

        running_loss = 0.0
        counter = 0
        # Iterate over data.
        for roll, pitch, frame in data_loader:
            frame = frame.to(device, dtype=torch.float)
            optimizer.zero_grad()
            loss = 0.0
            if args.roll:            
                roll = roll.to(device, dtype=torch.float)
                roll = normalize_roll(roll, roll_mean, roll_std)
                out_reg_roll, _ = model(frame) # i due outputs vengono rispettivamente da fc_reg e fc_cat
                roll = roll.view(roll.size(0), -1)
                loss_reg_roll = criterion_mse(out_reg_roll, roll)
                loss += loss_reg_roll ##CATEGORICAL LOSS DEACTIVATED

            if args.pitch:
                pitch = pitch.to(device, dtype=torch.float)
                pitch = normalize_pitch(pitch, pitch_mean, pitch_std)
                _, out_reg_pitch = model(frame) # i due outputs vengono rispettivamente da fc_reg e fc_cat
                pitch = pitch.view(roll.size(0), -1)
                loss_reg_pitch = criterion_mse(out_reg_pitch, pitch) ##loss MSE, prodotta con l' output singolo fc_reg della rete
                loss += loss_reg_pitch 

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter += 1

        ############ print the matrics ###########
       
        print('Training Mean Loss: {:.4f}'.format(running_loss/counter))
        model.eval()
        if args.testtrain:
            test_model(model,testtrain_dataset_loader,epoch)
        test_model(model,test_dataset_loader,epoch)
        torch.save(model.state_dict(), exper_path+weights_filename+"_epoch"+str(epoch)+".pth")

###LOADING THE PRETRAINED MODEL
model.load_state_dict(torch.load(args.pretrained,map_location=device))
model = model.to(device)

###MEASURING INFERENCE SPEED
model.eval()
model.to(device_cpu) ###DA TOGLIERE
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        frame_list = []
        for i in range(100): ##pre-loading 100 frames

            _, _, frame = test_dataset.__getitem__(i) 
            frame = frame.to(device_cpu, dtype=torch.float)
            frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
            frame_list.append(frame)

        ##warmp-up
        _, _, frame = test_dataset.__getitem__(100) 
        frame = frame.to(device_cpu, dtype=torch.float)
        frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
        out_reg_roll, out_reg_pitch = model(frame)

        start.record()
        for frame in frame_list: 
            out_reg_roll, out_reg_pitch = model(frame)
        end.record()
        torch.cuda.synchronize()
        print("Elapsed time (msec) for one image: ")
        print(start.elapsed_time(end)/100)


#### COMPRESSION AND FINETUNING
model.to(device)
model_stats = FlopCo(model, device = device)
compressor = CompressorVBMF(model,
                            model_stats,
                            ft_every=5, 
                            nglobal_compress_iters=1)
str_param = ""
if args.compression:
    str_param += "compressed"
    model.train()
    print("Compression and finetuning started!")
    while not compressor.done:
    # Compress layers
        try:
            compressor.compression_step()
        except Exception as e: 
            print(e)
        compressor.compressed_model = compressor.compressed_model.to(device)
        train_model(model=compressor.compressed_model, data_loader=train_dataset_loader, dataset_size=len(train_dataset_loader), optimizer=optimizer , scheduler=lr_sch, num_epochs=1)
    print("Compression ended! Measuring time performances:")
    model = compressor.compressed_model
    #model = model.to(device_cpu) ##DA TOGLIERE

###PRUNING
prune_flag= 0
model.eval()
if args.pruning:
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))
            prune_flag = 1
        elif isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))
            prune_flag = 1
    prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=args.pruning,)
    model = remove_parameters(model) 
if prune_flag:
    print("pruned!")
    str_param += "_pruned"


torch.save(model,args.model+"_"+str_param+".pth")

##futher training:
model = model.to(device)
model.train()
train_model(model=model, data_loader=train_dataset_loader, dataset_size=len(train_dataset_loader), optimizer=optimizer , scheduler=lr_sch, num_epochs=10)
torch.save(model,args.model+"_"+str_param+"_retrained.pth")


##CLOSING TENSORBOARD WRITER
writer.close()

