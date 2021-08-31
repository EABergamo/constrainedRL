import utils
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime
import torch.nn as nn
import numpy as np
import gym
import custom_planning
import torch
import os
import matplotlib.pyplot as plt

# Environment setup
env = gym.make('CustomPlanning-v0')

# GNN hyperparameters
degree = 5
state_dic_location = '/home/jcervino/summer-research/unlabelledPlanningML/experiments/flockingGNN-012-20210824141834/savedModels/LocalGNNArchitLast.ckpt'
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
n_agents = 12
n_features = 2 * (3 * degree + 2) + 1
sigma = 0.05

def main(n_episodes, do_print=True):
    reward_history = []
    collision_history = []
    goals_completed_history = []
    
    if (do_print):
        print('\tExecuting episodes...', end = ' ', flush = True)
    
    for episode in range(n_episodes):
        
        # Creates history and saves initial conditions,
        # the first additional dimension allows to use the GNN code.
        state_hist = np.zeros((1, t_samples, n_features, n_agents))        
        graph_hist = np.zeros((1, t_samples, n_agents, n_agents))
        prob_hist = np.zeros(t_samples)
        reward_ep = np.zeros(t_samples)
        collision_ep = np.zeros(t_samples)
        goals_completed = 0
        
        state_hist[0, 0, :, :], graph_hist[0, 0, :, :] = env.reset()

        
        for t in range(1, t_samples):
            x = torch.tensor(state_hist[:, 0:t, :, :])
            S = torch.tensor(graph_hist[:, 0:t, :, :]) 

            action, prob = utils.select_action(localGNN, x, S, sigma)
            prob_hist[t - 1] = prob
            
            state, reward, done, _ = env.step(action)
            state_hist[0, t, :, :] = state[0]
            graph_hist[0, t, :, :] = state[1]
            collision_ep[t] = state[2]
            
            reward_ep[t] = reward
            
            if done:
                break
            
            if (t == t_samples - 1):
                goals_completed = state[3]
            
        reward_history.append(np.mean(reward_ep))
        goals_completed_history.append(goals_completed)
        collision_history.append(np.sum(collision_ep))
        
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
        
        # Update GNN
        utils.update_policy(optim, 1, reward_ep, prob_hist)
        
    # Print
    if do_print:
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
        print("OK", flush = True)
    
    today = utils.save_model('constrainedRL', localGNN)
    
    return reward_history, collision_history, goals_completed_history, today
            
if __name__ == "__main__":
    episodes = 125000
    reward, collision, goals, today = main(episodes, do_print=True)
    
    figs, axs = plt.subplots(2, figsize=(12, 10))
    
    axs[0].plot(np.arange(episodes), goals)
    axs[0].set_ylabel('Goals Achieved')
    axs[0].set_xlabel('Episode')
    axs[0].set_title('Number of Goals Achieved')
    axs[0].grid()  
    
    axs[1].plot(np.arange(episodes), collision)
    axs[1].set_ylabel('Collisions')
    axs[1].set_xlabel('Episode')
    axs[1].set_title('Number of Collisions ($d_{min} \leq 0.75 \delta$)')
    axs[1].grid()  
          
    plt.savefig('/home/jcervino/summer-research/constrainedRL/experiments/constrainedRL/savedModels/lossPlots_' + today)