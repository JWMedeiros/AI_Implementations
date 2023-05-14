# AI for Breakout
# A Convolutional NN with A3C integrated, it also contains an LSTM for better predictions

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
# Need to set two standard deviations for the actor and the critic
def normalized_columns_initializer(weights, std = 1.0):
    #Creates a torch tensor of weights equal to number of weights
    out = torch.randn(weights.size())
    #Gets the weights of out, take sum of squared then square root it 
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # var(out) = std^2
    return out

# Function to initialize weights to make the learning optimal
def weights_init(m):
    # We are going to specifically initialize weights based off research papers
    classname = m.__class__.__name__
    # If we have a convolution connection
    if classname.find('Conv') != -1:
        # Run specific initialization of weights
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) #Dim1 x Dim2 x Dim3
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0] # Dim0 x Dim2 x Dim3
        w_bound = np.sqrt(6. / fan_in+fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)
        # Initialize Biases with 0s
        m.bias.data.fill_(0)
    # If we have a classic linear FC, there are less dimensions for weights
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in+fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)
        # Initialize Biases with 0s
        m.bias.data.fill_(0)

# Making the A3C Brain
# It will have have eyes, fully connected layers and memory
# The most powerful AI model up to this date
class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        #Making the eyes of the AI using standard architecture
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        #Creating the memory of the brain, its used to learn the temporal properties of the inputs
        # The first arg is the result of the count neurons function in doom, that ends up being the end result
        self.lstm = nn.LSTMCell(32*3*3, 256)
        #Get number of possible actions
        num_outputs = action_space.n
        #Create connection for action and for critic, value of the v function from the state
        #This is shared amongst all the actors
        self.critic_linear = nn.Linear(256, 1) # OP = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) #OP is Q,S,A
        #Initialize weights for object
        self.apply(weights_init)
        #Set different variance (Standard deviation) for actor and critic
        # This helps manage eploration vs eploitation
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        #self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()
        
    def forward(self, inputs):
        inputs, (hx,cx) = inputs
        #Using the Elu activation function
        x = F.elu(self.conv1(inputs))
        #Propogate the signals
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        #Flatten x
        x = x.view(-1,32*3*3)
        #LSTM
        (hx,cx) = self.lstm(x, (hx,cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx,cx)