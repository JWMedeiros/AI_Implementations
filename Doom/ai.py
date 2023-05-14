# AI for Doom
#This code requires some additional dependancies to work, found on the google Collab link.
#Namely installing all the libraries related to the doom environment


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
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,256,256)), out_features = 40)
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
class AI():
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    def __call__(self, inputs):
        #Recieving the input images from the brain to enter the neural network
        #Maxing a torch variable for later computing the SGD
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        #Then propogate images through the eyes of the AI using the brains forward function
        output = self.brain(input)
        #And lastly uses the body's forward function to turn the q values into actions
        actions = self.body(output)
        #Convert actions back and reformate them
        return actions.data.numpy()
        


# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
#These lines are from the Open AI gym tutorials for getting the Doom space and number of actions
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width=80, height=80, greyscale=True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

#Building an AI
#Create both the brain and the body then the ai from them
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

#Setting up Experience Replay with number of steps and memory capacity
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

#Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    #Calculate cumulative reward over n steps
    for series in batch:
        #Input of first and last signal of batch
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        #Compute cumulative reward using N-Step q learning
        #Cum reward is 0 if the last transition is done, otherwise its the max of the q values
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        #From second last step to first step
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + cumul_reward*gamma
        #Prepare inputs and targets
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        #We only need 1 input and 1 state because we learn from the next 10 steps generated from that state
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

#Making the moving average on 100 steps
class MA():
    def __init__(self, size):
        self.list_list_of_rewards = []
        self.size = size
        
    #Add cumulative reward to list of rewards
    def add(self, rewards):
        #If the rewards are a list,
        if isinstance(rewards, list):
            self.list_list_of_rewards += rewards
        else:
            self.list_list_of_rewards.append(rewards)
        #List of rewards never has more than size elements
        while len(self.list_list_of_rewards) > self.size:
            del self.list_list_of_rewards[0]
    
    #Compute the average of list of rewards
    def average(self):
        return np.mean(self.list_list_of_rewards)
    
ma = MA(100)

#Training the AI
#Making the loss function and then the optimizer afterwards
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100
#Starting the main training loop
for epoch in range(1, nb_epochs+1):
    #Each epoch is 200 runs of 10 steps
    memory.run_steps(200)
    #32 is common, but for our purposes its better to do more
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        #Convert inputs and targets to torch variables
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        #Calculate loss error
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        #Backpropogate loss error
        loss_error.backward()
        #Update weights with SGD
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))