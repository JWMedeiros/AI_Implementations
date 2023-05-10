# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN,self).__init__()
        #Applies convolution to first set of images
        #Input is 1 image, output is 32 preprocessessed images with detected features 
        #using the dimensions of the square that will go thru the original image
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=2)
        #The dimensions of the doom image are black and white, 80x80 images
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    
    #Function to count the number of needed neurons from the incoming images dimensions
    def count_neurons(self, img_dim):
        #Creating random image based off image dimensions
        x = Variable(torch.rand(1,*img_dim))
        x = self.propogate_neurons(x)
        #signals now propogated to the third convolutional layer and now must be flattened
        return x.data.view(1,-1).size(1)
    
    def forward(self,x):
        x = self.propogate_neurons(x)
        #Flattening the convolution layer composed of several channels using Pytorch
        x = x.view(x.size(0), -1)
        #Connecting the flattening layer to the hidden layer using the rectifier function
        x = F.relu(self.fc1(x))
        #Propogate signal from hidden to OP layer
        x = self.fc2(x)
        return x
    
    def propogate_neurons(self,x):
        #activating the neurons using max pooling with the convoluted image, kernal size of 3 and sliding window size of 2
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x
        
# Making the Body
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T=T
        
    def forward(self, outputs):
        #Taking the outputs from the brain to play the right action using Softmax Technique
        probs = F.softmax(outputs * self.T)
        #Get action sample from distribution of probabilities
        actions = probs.multinomial()
        return actions

# Making the AI


# Part 2 - Training the AI with Deep Convolutional Q-Learning
