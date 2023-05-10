# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the Neural Network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action, hidden_layers):
        super(Network,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #The next two lines are connecting up the neurons
        #Number of input neurons, and number of neurons you want for hidden layer, 30 is a good starting point
        self.fc1 = nn.Linear(input_size, hidden_layers)
        #Connectiing the hidden layer to OP
        self.fc2 = nn.Linear(hidden_layers, nb_action)
    
    #Activates the neurons, and returns the q values for each action
    def forward(self, state):
        #Rectifier function applied from input to hidden layer
        x = F.relu(self.fc1(state))
        #Generate q values from the result of applying the above
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience Replay
class ReplayMemory(object):
    
    #Initialize with a capacity of memories and an empty memory list
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    #Push function adds memories, and deletes them once they get over capacity
    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #Takes random samples from memory that have a fixed size of batch size, and formats them with zip
        samples = zip(*random.sample(self.memory, batch_size))
        #Returns the samples as pyTorch variables
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Complete implementation of Deep Q Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, hidden_layers, gamma):
        self.gamma = gamma
        self.reward_window = []
        #Creation of Neural Network object
        self.model = Network(input_size, nb_action, hidden_layers)
        #Creation of ReplayMemory object using 100k samples
        self.memory = ReplayMemory(100000)
        #Connects the Pytorch Adam optimizer to the neural network with a learning rate that isnt too high so it can explore
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        #Using Softmax function, changing the tensor into a variable which saves memory and improves performance
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) #Temperature variable = to 7 (Affects probability of picking best q value)
        #Selects action from probability distribution, must be converted back from Pytorch
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    #Forward and backwards propogation function
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #Get the best actions from all the states
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #Get max of the values of q states from the previous states
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        #Using hoober function for loss calculation in Q learning
        td_loss = F.smooth_l1_loss(outputs, target)
        #Re-initialize the optimizer
        self.optimizer.zero_grad()
        #Backpropogate error into the Neural Network
        td_loss.backward(retain_graph = True)
        #Update weights
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        #Update when we reach the new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #Add transition event to Memory with some Pytorch conversions
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #Play an action after reaching the new state
        action = self.select_action(new_state)
        #Learn after selecting an action using 100 transitions
        if len(self.memory.memory)>100:
            #This line is reversed action and reward due to sample for some reason
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #Update last state and last action and last reward and the reward window to see how training is going
        self.last_action = action
        self.last_state=new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window)>1000:
            del self.reward_window[0]
        return action
    
    #Returns the average of all the rewards in the reward window +1
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    #Save function that saves the brain, both neural network and optimizer
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
     
    #Load function that loads the brain back into 
    def load(self):
        if (os.path.isfile('last_brain.pth')):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            print('no checkpoint found...')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        