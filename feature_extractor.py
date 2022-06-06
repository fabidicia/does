import torch
from torch import optim, nn
from torchvision import models, transforms

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 



    # class FeatureExtractor(nn.Module):
    # def __init__(self):
    #     super(FeatureExtractor, self).__init__()
    #     self.net = models.googlenet(pretrained=True)
    #     # If you treat GooLeNet as a fixed feature extractor, disable the gradients and save some memory
    #     for p in self.net.parameters():
    #         p.requires_grad = False
    #     # Define which layers you are going to extract
    #     self.features = nn.Sequential(*list(self.net.children())[:4])

    # def forward(self, x):
    #     return self.features(x)