# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    #Desyncronize each training agent by shifting the seed with rank
    torch.manual_seed(params.seed + rank)
    #Get the environment
    env = create_atari_env(params.env_name)
    #Align the seed of the env with the each agent
    env.seed(params.seed+ rank)
    #Get the A3C brains
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    #Prepare input states (Images)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    #Initialize the length of 1 episode
    episode_length = 0
    while True:
        episode_length+=1
        #Sync with shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        #Training Process
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            #Get the predictions from the model
            value, action_values, (hx,cx) = model((Variable(state.unsqueeze(0)),(hx,cx)))
            #Use Softmax to play action
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            #Get entropy
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            #Take a random draw based on softmax probs
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            #Play action using environment
            state,reward, done = env.step(action.numpy())
            #Make sure actor doesnt get stuck and game finishes
            done = (done or episode_length>=params.max_episode_length)
            #Clamp reward between -1 and 1
            reward = max(min(reward,1),-1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        #Update rewards for shared network
        R = torch.zeroes(1,1)
        if not done:
            value,_,_, = model((Variable(state.unsqueeze(0)),(hx,cx)))
            R = value.data
        values.append(Variable(R))
        #Initialize losses
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeroes(1,1) #A(a,s) = Q(a,s) - V(s)
        #Compute losses
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i] #R = r_0 + gamma * r_1 + gamma^2 * r_2 +... gamma^(n-1) * r_(n-1)+gamma^nb_steps * V(last_state)
            advantage = R -values[i]
            value_loss = value_loss+0.5* advantage.pow(2) # Q*(a*,s) = V*(s)
            TD = rewards[i]+params.gamma *values[i+1].data - values[i].data
            gae = gae* params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i)
            policy_loss = policy_loss - log_probs[i]* Variable(gae)- 0.01 * entropies[i]# policy_loss = -sum_i log(pi_i)*gae +0,01*H_i
        #Perform SGD with twice as much emphasis on policy loss as it is smaller
        optimizer.zero_grad()
        (policy_loss+ 0.5*value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model,shared_model)
        optimizer.step()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        