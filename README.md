# DOES - Deep Orientation (of roll and pitch) Estimation at Sea
 <!-- ## Code from the paper "DOES: A Deep Learning-based approach to estimate roll and pitch at sea."... -->

All the instructions to correctly run DOES can be found in the [instruction.md](./instruction.md) file.

<!-- If you wish to try DOES on a custom dataset, keep in mind that probably a basic finetuning of the model weights will be necessary to obtain good results! For this reason, we suggest to make a first training on the ROPIS datasets (also to further verify if it correctly runs) *having prior modified the script to save and load the weights*. At this point, you can proceed with the tuning. ... -->

# DOES - Overview
DOES is an affordable and smart supportive Deep Learning model which can be combined to the actual IMU-based configuration to improve the accuracy of the attitude estimations. 
It is able to output the correct vehicle attitude (in terms of roll and pitch angles) by processing the sea horizon view recorded by a low-cost camera.
The model has in fact been trained on the the first release of the ***ROPIS*** dataset, which has been created through ***FrameWO***, an Android application developed for the scope.

