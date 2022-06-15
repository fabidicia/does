################################# Various import #######################################

import numpy as np
from random import randint
import torch
from torchvision import transforms, models
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from new_dataset import HorizonDataset
from networks import Net
import argparse
from timeit import default_timer as timer
import time
############################# Code call definition ######################################

parser = argparse.ArgumentParser("script to train horpe on oneplus videoframe")
parser.add_argument('--roll', type=int, default=1)
parser.add_argument('--pitch', type=int, default=1)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--test_hld', action="store_true")
parser.add_argument('--gpu', type=str,default='True')
parser.add_argument('--onnx',type=str, default='cuda')

args = parser.parse_args()

############################# Image and data -related parameters definitions ######################################

if args.gpu=='True':
    device = torch.device('cuda') 
    model = Net(args)

    ###MEASURING INFERENCE SPEED
    model.eval()
    model.to(device) ###DA TOGLIERE

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            frame_list = []
            for i in range(100): ##pre-loading 100 frames
                frame = torch.rand(3,256,256) 

                frame = frame.to(device, dtype=torch.float)
                frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
                frame_list.append(frame)

            ##warmp-up
            frame = torch.rand(3,256,256) 
            frame = frame.to(device, dtype=torch.float)
            frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
            out_reg_roll, out_reg_pitch = model(frame)

            start.record()
            for frame in frame_list: 
                out_reg_roll, out_reg_pitch = model(frame)
            end.record()
            torch.cuda.synchronize()
            print("Elapsed time (msec) for one image: ")
            print(start.elapsed_time(end)/len(frame_list))
elif args.gpu=='False':
    device = torch.device('cpu') 
    model = Net(args)

    ###MEASURING INFERENCE SPEED
    model.eval()
    model.to(device) ###DA TOGLIERE

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            frame_list = []
            for i in range(10): ##pre-loading 100 frames
                frame = torch.rand(3,256,256) 

                frame = frame.to(device, dtype=torch.float)
                frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
                frame_list.append(frame)

            ##warmp-up
            frame = torch.rand(3,256,256) 
            frame = frame.to(device, dtype=torch.float)
            frame = frame.unsqueeze(0) ##aggiungo una dimensione per matchare la shape di outputs!
            out_reg_roll, out_reg_pitch = model(frame)

            start.record()
            for frame in frame_list: 
                out_reg_roll, out_reg_pitch = model(frame)
            end.record()
            torch.cuda.synchronize()
            print("Elapsed time (msec) for one image: ")
            print(start.elapsed_time(end)/len(frame_list))
else:
    device = torch.device('cuda') 
    model = Net(args)

    ###MEASURING INFERENCE SPEED
    model.eval()
    model.to(device) ###DA TOGLIERE

print("Starting onnx run...")
import torch.onnx
batch_size= 1
input = torch.randn(batch_size,3,256,256) 
input = input.to(device, dtype=torch.float)
out_reg_roll, out_reg_pitch = model(input)

# Export the model
torch.onnx.export(model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,
                  verbose=False,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output1','output2'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output1' : {0 : 'batch_size'},
                                'output2' : {0 : 'batch_size'}})
del model
torch.cuda.empty_cache()
import onnx,onnxruntime
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
providers_dict = {'cuda':'CUDAExecutionProvider', 'rt': 'TensorrtExecutionProvider'}
ort_session = onnxruntime.InferenceSession("model.onnx",providers=[providers_dict[args.onnx]])
#so = ort.SessionOptions()
#so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#ort_session = onnxruntime.InferenceSession(model_name, so)
output1,output2 = ort_session.run(
    None,
    {"input": np.random.randn(10, 3, 256, 256).astype(np.float32)},)

start = timer()
n_times = 100
dummy_list = []
for i in range(n_times):
    output1,output2 = ort_session.run(
    None,
    {"input": np.random.randn(1, 3, 256, 256).astype(np.float32)},)
    dummy_list.append(output1-output2)
elapsed = timer() - start
print("Elapsed time in msec, ONNX format:")
print(elapsed/n_times)
