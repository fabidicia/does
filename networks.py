################################# Various import #######################################

import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


out_features_mse = 1

class Net(nn.Module):
    def __init__(self, args, out_features_categorical=180):
        super(Net, self).__init__()
        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = None

        if args.model == "vgg19bn":
            model = models.vgg19_bn(pretrained=True)
            modules = list(model.children())
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.classifier[6] = nn.Identity()
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=4096, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=4096, out_features=out_features_mse)
            )            # self.fc_cat_roll = nn.Sequential( ## fully connected regression branch

        if args.model == "vgg19":
            model = models.vgg19(pretrained=True)
            modules = list(model.children())
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.classifier[6] = nn.Identity()
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=4096, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=4096, out_features=out_features_mse)
            )


        if args.model == "densenet161":
            model = models.densenet161(pretrained=True)
            modules = list(model.children())
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.classifier = nn.Identity()
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=2208, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=2208, out_features=out_features_mse)
            )


        if args.model == "mobilenet_large":
            model = models.mobilenet_v3_large(pretrained=True)
            modules = list(model.children())
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.classifier[3] = nn.Identity()
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=1280, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=1280, out_features=out_features_mse)
            )


        if args.model == "mobilenet_small":
            model = models.mobilenet_v3_small(pretrained=True)
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.classifier[3] = nn.Identity()
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=1024, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch
               nn.Linear(in_features=1024, out_features=out_features_mse)
            )


        if args.model == "resnet18":
            model = models.resnet18(pretrained=True)
#            modules = list(model.children())[:-1]
            self.model = model #nn.Sequential(*modules) #rebuilding the network after fc removal
            self.model.fc = nn.Identity()

            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=512, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=512, out_features=out_features_mse)
            )

        elif args.model == "resnet152":
            model = models.resnet152(pretrained=True)
            modules = list(model.children())[:-1] #with this trick I remove model.fc
            self.model = nn.Sequential(*modules) #rebuilding the network after fc removal
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=2048, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=2048, out_features=out_features_mse)
            )

        elif args.model == "resnet50":
            model = models.resnet50(pretrained=True)
            modules = list(model.children())[:-1] #with this trick I remove model.fc
            self.model = nn.Sequential(*modules) #rebuilding the network after fc removal
            self.fc_reg_roll = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=2048, out_features=out_features_mse)
            )
            self.fc_reg_pitch = nn.Sequential( ## fully connected regression branch,
                        nn.Linear(in_features=2048, out_features=out_features_mse)
            )


    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0],-1) 
        if self.dropout:
            x = self.dropout(x)
        x_reg_roll = self.fc_reg_roll(x)
        x_reg_pitch = self.fc_reg_pitch(x)

        return x_reg_roll,x_reg_pitch


