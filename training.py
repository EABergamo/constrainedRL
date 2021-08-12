import utils
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime
import torch.nn as nn
import numpy as np
import gym
import custom_planning
import torch

# Environment setup
env = gym.make('CustomPlanning-v0')

# GNN hyperparameters
degree = 2
state_dic_location = '/home/jcervino/summer-research/unlabelledPlanningML/experiments/flockingGNN-003-20210805130549/savedModels/LocalGNNArchitLast.ckpt'
dimNodeSignals = [2*(3 * degree + 2)+1, 64]
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

localGNN.load_state_dict(torch.load(state_dic_location, map_location=torch.device('cpu')))

# Optimizer hyperparameter 
learningRate = 0.0005 
beta1 = 0.9 
beta2 = 0.999 

optim = optim.Adam(localGNN.parameters(),
                    lr = learningRate,
                    betas = (beta1, beta2))

# Simulation hyperparameters
t_samples = 30
n_agents = 3
n_features = 2 * (3 * degree + 2) + 1

def main(n_episodes, do_print=True):
    reward_history = []
    
    if (do_print):
        print('\tExecuting episodes...', end = ' ', flush = True)
    
    for episode in range(n_episodes):
        
        # Creates history and saves initial conditions,
        # the first additional dimension allows to use the GNN code.
        state_hist = np.zeros((1, t_samples, n_features, n_agents))        
        graph_hist = np.zeros((1, t_samples, n_agents, n_agents))
        
        state_hist[0, 0, :, :], graph_hist[0, 0, :, :] = env.reset()
        reward_ep = 0
        
        for t in range(1, t_samples):
            x = torch.tensor(state_hist[:, 0:t, :, :])
            S = torch.tensor(graph_hist[:, 0:t, :, :]) 

            with torch.no_grad():
                action = localGNN(x, S)
                action = action.numpy()
                action = action[0, -1].T
            
            state, reward, done, _ = env.step(action)
            state_hist[0, t, :, :] = state[0]
            graph_hist[0, t, :, :] = state[1]
            
            reward_ep = reward_ep + (reward - reward_ep) / t
            
            if done:
                break
            
        reward_history.append(reward_ep)
        
        if do_print:
            percentageCount = int(100 * episode + 1) / n_episodes
            if episode == 0:
                # It's the first one, so just print it
                print("%3d%%" % percentageCount,
                    end = '', flush = True)
            else:
                # Erase the previous characters
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                    end = '', flush = True)
        
        # TODO: Update GNN
        
    # Print
    if do_print:
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
        print("OK", flush = True)
    
            
if __name__ == "__main__":
    episodes = 5000
    main(episodes)