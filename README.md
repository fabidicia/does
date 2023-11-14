# DOES - Deep Orientation (of roll and pitch) Estimation at Sea
Code from the paper "DOES: A Deep Learning-based approach to estimate roll and pitch at sea.". 

This directory also includes the model optimization techniques tested for embedded devices presented in the conference paper "Deep models optimization on embedded devices to improve the orientation estimation task at sea".

All the instructions to correctly run DOES can be found in the [instruction.md](./instruction.md) file.
If you wish to try DOES on a custom dataset, keep in mind that probably a basic finetuning of the model weights will be necessary to obtain good results! For this reason, we suggest to make a first training on the ROPIS datasets (also to further verify if it correctly runs) *having prior modified the script to save and load the weights*. At this point, you can proceed with the tuning.

If you liked our work and you wish to use it in your research, please consider to cite us!

### Original Article
> Di Ciaccio, F., Russo, P., & Troisi, S. (2022). Does: A deep learning-based approach to estimate roll and pitch at sea. IEEE access, 10, 29307-29321.

> @article{di2022does,
  title={Does: A deep learning-based approach to estimate roll and pitch at sea},
  author={Di Ciaccio, Fabiana and Russo, Paolo and Troisi, Salvatore},
  journal={IEEE access},
  volume={10},
  pages={29307--29321},
  year={2022},
  publisher={IEEE}
}

### Conference paper
> Russo, P., & Di Ciaccio, F. (2022, October). Deep models optimization on embedded devices to improve the orientation estimation task at sea. In 2022 IEEE International Workshop on Metrology for the Sea; Learning to Measure Sea Health Parameters (MetroSea) (pp. 44-49). IEEE.

> @inproceedings{russo2022deep,
  title={Deep models optimization on embedded devices to improve the orientation estimation task at sea},
  author={Russo, Paolo and Di Ciaccio, Fabiana},
  booktitle={2022 IEEE International Workshop on Metrology for the Sea; Learning to Measure Sea Health Parameters (MetroSea)},
  pages={44--49},
  year={2022},
  organization={IEEE}
}

# DOES - Overview
DOES is an affordable and smart supportive Deep Learning model which can be combined to the traditional IMU-based configurations to improve the accuracy of the attitude estimation. 
It is able to output the correct vehicle attitude (in terms of roll and pitch angles) by processing the sea horizon view recorded by a low-cost camera.
The model has in fact been trained on the the first release of the ***ROPIS*** dataset, which has been created through ***FrameWO***, an Android application developed for the scope.

Further studies have been made to improve the performances of DOES for its deployment on embedded systems, using lighter backbone architectures and model optimization techniques.
<!--- ### DANAE Roll estimation - OXIO Dataset

![plot](./Results_Figure/oxford_LKF_phi.jpg)
![plot](./Results_Figure/oxford_danae1_phi.jpg)

### DANAE Pitch estimation - UCS Dataset
![plot](./Results_Figure/ucs_lkf_theta.jpg)
![plot](./Results_Figure/ucs_danae1_theta.jpg)

DANAE++ is the enhanced version of the first architecture: it is able to denoise IMU/AHRS data obtained through both the Linear (LKF) and Extended (EKF) Kalman filter-derived values. Better results are achieved by DANAE++ also when compared to common low-pass filters (in our study, the [Butter LP filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
) and the [Uniform1d filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html) both provided by the Scipy library).

The following images shows the results obtained by DANAE++ w.r.t. the roll angle estimation provided by the EKF and the LP filters for the OXIO Dataset, together with DANAE++ performance on the pitch angle estimation for the UCS Dataset.

### DANAE++ Roll estimation - OXIO Dataset
![plot](./Results_Figure/oxford_EKF_phi.jpg)
![plot](./Results_Figure/oxford_danae++_phi.jpg)
![plot](./Results_Figure/comparative_filters_butter_phi.jpg)
![plot](./Results_Figure/comparative_filters_uniform_phi.jpg)

### DANAE++ Pitch estimation - UCS Dataset
![plot](./Results_Figure/ucs_ekf_theta.jpg)
![plot](./Results_Figure/ucs_danae++_theta.jpg) ... -->

# References
[Original Article]: https://ieeexplore.ieee.org/document/9733371 published by IEEE, IEEE Access, 2022

[Conference Paper]: https://ieeexplore.ieee.org/document/9950745 published by IEEE, 2022 IEEE International Workshop on Metrology for the Sea; Learning to Measure Sea Health Parameters (MetroSea)

