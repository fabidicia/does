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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
torch.backends.cudnn.benchmark = True
from new_dataset import HorizonDataset
from tqdm import tqdm
import cv2 
import two_objectives_horizon_detection as tohd
from scipy.optimize import lsq_linear
from torch.nn.utils import prune
#################################### functions ##########################

# https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
from tqdm.auto import tqdm
from joblib import Parallel

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def ploton_tensorboard(writer, image, Datas=None, Lines=None, Labels=None, Name="Image", xlabel=None, ylabel=None,ylim=None):
    rnd = randint(0,10000)
    # with open(Name+".pkl", "wb") as open_file:
    #     pickle.dump(Datas, open_file)
    if Name == "Image":
         Name = Name + str(rnd)
    plt.figure(rnd,figsize=(14,7))
        # code to plot the image in tensorboard
    plt.title(Name)
    if ylabel is not None:
         plt.ylabel(ylabel)
    if ylabel is not None:
        plt.xlabel(xlabel)
    if ylim is not None:
        axes = plt.gca()
        axes.set_ylim(ylim)
    #times = [i for i in range(len(Datas[0]))]
    # for data,line,label in zip(Datas, Lines, Labels):
    #     plt.plot(times, data, line,label=label)
    #     plt.legend(loc="lower right")
    # plt.imshow(image)
    # plt.show()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    #image = PIL.Image.open(image)
    #image = ToTensor()(image)
    writer.add_image(Name, image, 0)
    # plt.clf()
    # plt.cla()
    # plt.close()
    writer.close()

def normalize_roll(roll, roll_mean, roll_std):
    return (roll - roll_mean) / roll_std  #substract the mean, divide by std, so that now it has mean 0 std 1

def normalize_pitch(pitch, pitch_mean, pitch_std):
    return (pitch - pitch_mean) / pitch_std 

def denormalize_roll(roll, roll_mean, roll_std):
    return (roll * roll_std) + roll_mean   #multiply by std, add the mean

def denormalize_pitch(pitch, pitch_mean, pitch_std):
    return (pitch * pitch_std) + pitch_mean   #multiply by std, add the mean

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

########################## Hopenet functions for the optimizer definition #################

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

######################### Dataset elaboration #################################

if __name__ == '__main__':

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

#https://github.com/sallamander/horizon-detection
#It is based on Otzu method: A threshold selection method from gray-level histograms 
#https://cw.fel.cvut.cz/b201/_media/courses/a6m33bio/otsu.pdf
def detect_horizon_line(image_grayscaled):
    """Detect the horizon's starting and ending points in the given image

    The horizon line is detected by applying Otsu's threshold method to
    separate the sky from the remainder of the image.

    :param image_grayscaled: grayscaled image to detect the horizon on, of
     shape (height, width)
    :type image_grayscale: np.ndarray of dtype uint8
    :return: the (x1, x2, y1, y2) coordinates for the starting and ending
     points of the detected horizon line
    :rtype: tuple(int)
    """

    msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
           'of shape (height, width).')
    assert image_grayscaled.ndim == 2, msg
    image_blurred = cv2.GaussianBlur(image_grayscaled, ksize=(3, 3), sigmaX=0)

    _, image_thresholded = cv2.threshold(
        image_blurred, thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    image_thresholded = image_thresholded - 1
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                    kernel=np.ones((9, 9), np.uint8))

    horizon_x1 = 0
    horizon_x2 = image_grayscaled.shape[1] - 1
    horizon_y1 = max(np.where(image_closed[:, horizon_x1] == 0)[0])
    horizon_y2 = max(np.where(image_closed[:, horizon_x2] == 0)[0])

    return horizon_x1, horizon_x2, horizon_y1, horizon_y2

def hdl_predict(image):
    image_grayscale = np.array(image.convert('L'))
    try:
        x1,x2,y1,y2 = detect_horizon_line(image_grayscale)
    except:
        return 0.0, image_grayscale.shape[0]/2.0 #default line, horizontal, positioned at half image
    angle = np.arctan2(y2-y1,x2-x1) * 360 / (2*np.pi)    
    return angle,y2-y1

def line(m, b, x, y):
	return y - m*x - b

#https://raw.githubusercontent.com/k29/horizon_detection/master/myHorizon.py
#https://github.com/k29/horizon_detection
#Vision-Guided Flight Stability and Control for Micro Air Vehicles, Scott M. Ettinger https://www.tandfonline.com/doi/abs/10.1163/156855303769156983
def detectHorizon(cvImg):
    xSize =cvImg.shape[1]
    ySize = cvImg.shape[0]
        #keeping a resolution of 50 to generate the values of slope and y intercept
       #resolution can be changed by changing this value
    res = 100.0
    slope = np.linspace(-1,1,int(res))
    inter = np.linspace(0,ySize,int(res))
    maximum = []
    J_max = 0
        #iterate over all the slope and intercept values
    for m in range(len(slope)):
        for b in range(len(inter)):
            sky = [] #array of pixel values containing sky
            gnd = [] #array of pixel values containing ground
            #iterate over all the pixels in the image and add them to sky and gnd
            for i in range(xSize):
                for j in range(ySize):
                    if((line(slope[m],inter[b],i,j)*(-1*inter[b])) > 0):
                        sky.append(cvImg[j,i])
                    else:
                        gnd.append(cvImg[j,i])
			#find covariance of the sky and gnd pixels

            sky = np.transpose(sky)
            gnd = np.transpose(gnd)
            try:
                co_s = np.cov(sky)
                co_g = np.cov(gnd)
                co_sD = np.linalg.det(co_s)
                co_gD = np.linalg.det(co_g)
                eig_s, _ = np.linalg.eig(co_s)
                eig_g, _ = np.linalg.eig(co_g)
                J = 1/(co_sD + co_gD + (eig_s[0]+eig_s[1]+eig_s[2])**2 + (eig_g[0]+eig_g[1]+eig_g[2])**2)
                if J > J_max:
                    J_max = J
                    maximum = [slope[m], inter[b]]
                    print(maximum)
            except Exception:
                import pdb; pdb.set_trace()
    return maximum[0],maximum[1]

def hdl_predict2(image):
    image_grayscale = np.array(image)
    image_grayscale = cv2.resize(image_grayscale, (0,0), fx=1/10, fy=1/10)
    slope = detectHorizon(image_grayscale)
    angle = slope
    return angle

#https://github.com/citrusvanilla/horizon_detection
def hdl_predict3(image):
    image_grayscale = np.array(image.convert('L'))

    global_img_reduction = 0.05
    global_angles = (-42,40,2)
    global_distances = (10,90,5) 
    global_buffer_size = 3
#GLOBAL OBJECTIVE MAIN ROUTINE
    global_search = tohd.main(image_grayscale, 
                          img_reduction = global_img_reduction,
                          angles = global_angles, 
                          distances = global_distances, 
                          buffer_size = global_buffer_size, 
                          local_objective = 0)  #2m5s

#GLOBAL OBJECTIVE OPTIMIZATION SURFACE
    objective_1 = np.max((global_search[:,:,0] - global_search[:,:,1]),0) / (global_search[:,:,2])
    above_two_sigma = (2* np.nanstd(objective_1)) + np.nanmean(objective_1)
    local_angles = global_search[np.where(objective_1 > above_two_sigma)[0],np.where(objective_1 > above_two_sigma)[1]][:,6]
    local_distances = global_search[np.where(objective_1 > above_two_sigma)[0],np.where(objective_1 > above_two_sigma)[1]][:,7]
    try:
        local_angle_range = (int(np.min(local_angles))-2,int(np.max(local_angles))+3,1)
        local_distance_range = (int(np.min(local_distances))-2,int(np.max(local_distances))+3,1)
    except:
        local_angle_range = (-42,+40,1)
        local_distance_range = (10,+90,1)

    local_img_reduction = 0.125
    local_angles = local_angle_range
    local_distances = local_distance_range
    local_buffer_size = 5
    try:
        local_search = tohd.main(image_grayscale,
                         img_reduction = local_img_reduction,
                         angles = local_angles, 
                         distances = local_distances, 
                         buffer_size = local_buffer_size, 
                         local_objective = 1)  #2m5s
        objective_2 =  (local_search[:,:,4] - local_search[:,:,5])**2 / local_search[:,:,2]
        horizon_line = local_search[np.unravel_index(objective_2.argmax(), objective_2.shape)]
    except:
        return 0.0,0.0
    line_coordinates = tohd.get_plane_indicator_coord(image_grayscale,int(horizon_line[6]),horizon_line[7]/100,0)[2:4]
    
    return horizon_line[6], abs(line_coordinates[1][1] - line_coordinates[0][1])

##Linear Least Squares calibration; it returns the calibrated pred:
def LLSc(gt,pred):
    num=len(gt)
    A = np.zeros((num,2),dtype=np.float)
    A[:,0] = pred
    A[:,1] = 1.0
    b = np.zeros((num,),dtype=np.float)
    b[:] = gt
    res = lsq_linear(A, b, verbose=0)
    tmp = np.array(pred)*res.x[0]+res.x[1]
    return tmp.tolist()

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

#to calculate numbers of parameters
#sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_total_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pruned_parameters_count(pruned_model):
    params = 0
    for param in pruned_model.parameters():
        if param is not None:
            params += torch.nonzero(param).size(0)
    return params

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(image, padding, 0, 'constant')
