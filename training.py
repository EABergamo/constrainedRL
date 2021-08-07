import utils
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime
import torch.nn as nn
import numpy as np
import gym

# Environment setup
env = gym.make('UnlabelledPlanning')

# GNN hyperparameters
state_dic_location = ''
dimNodeSignals = [2*(3 * 3 + 2)+1, 64]
nFilterTaps = [3]
bias = True
nonlinearity = nn.Tanh
dimReadout = [2]
dimEdgeFeatures = 1

localGNN = architTime.LocalGNN_DB(dimNodeSignals, 
                                  nFilterTaps, 
                                  bias, 
                                  nonlinearity, 
                                  dimReadout, 
                                  dimEdgeFeatures)

localGNN.load_state_dict(state_dic_location)

# Optimizer hyperparameter 
learningRate = 0.0005 
beta1 = 0.9 
beta2 = 0.999 

optim = optim.Adam(localGNN.parameters(),
                    lr = learningRate,
                    betas = (beta1, beta2))

# Simulation hyperparameters
t_samples = 30
episodes = 50000
n_agents = 50
degree = 3
n_features = 2 * (3 * degree + 2) + 1

def main(episodes):
    reward_history = []
    
    for episode in range(episodes):
        
        # Creates history and saves initial conditions
        state_hist = np.zeros((t_samples, n_agents, n_features))        
        graph_hist = np.zeros((t_samples, n_agents, n_agents))
        
        state_hist[0, :, :], graph_hist[0, :, :] = env.reset()
        reward_ep = 0
        
        for t in range(1, t_samples):
            action = localGNN(state_hist[0:t, :, :], graph_hist[0:t, :, :])
            action = action.numpy()
            action = action[-1]
            
            state, reward, done, _ = env.step(action)
            state_hist[t, :, :] = state
            graph_hist[t, :, :] = env.Graph
            
            reward_ep = reward_ep + (reward - reward_ep) / t
            
            if done:
                break
            
        reward_history.append(reward_ep)
        # TODO: Update GNN
            
    
            
        