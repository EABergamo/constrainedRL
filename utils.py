import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
from collections import deque 
import os

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime

def compute_agents_initial_positions(n_agents, comm_radius,
                                    min_dist = 0.1, doPrint=False, **kwargs):
    """ 
    Generates a NumPy array with the 
    initial x, y position for each of the n_agents

    Parameters
    ----------
    n_agents : int
        The total number of agents that will take part in the simulation
    n_samples : int
        The total number of samples.
    comm_radius : double (legacy code)
        The communication radius between agents (determines initial spacing between agents) 
    min_dist : double
        The minimum distance between each agent

    Returns
    -------
    np.array (n_samples x n_agents x 2) 
    """
    
    zeroTolerance=1e-9
    
    if (doPrint):
        print('\tComputing initial positions matrix...', end = ' ', flush = True)
    
    assert min_dist * (1.+zeroTolerance) <= comm_radius * (1.-zeroTolerance)
    
    min_dist = min_dist * (1. + zeroTolerance)
    comm_radius = comm_radius * (1. - zeroTolerance)
    
        
    # This is the fixed distance between points in the grid
    distFixed = (comm_radius + min_dist)/(2.*np.sqrt(2))
    
    # This is the standard deviation of a uniform perturbation around
    # the fixed point.
    distPerturb = (comm_radius - min_dist)/(4.*np.sqrt(2))
    
    # How many agents per axis
    n_agentsPerAxis = int(np.ceil(np.sqrt(n_agents)))
    
    axisFixedPos = np.arange(-(n_agentsPerAxis * distFixed)/2,
                                (n_agentsPerAxis * distFixed)/2,
                                step = distFixed)
    
    # Repeat the positions in the same order (x coordinate)
    xFixedPos = np.tile(axisFixedPos, n_agentsPerAxis)

    # Repeat each element (y coordinate)
    yFixedPos = np.repeat(axisFixedPos, n_agentsPerAxis)
    
    # Concatenate this to obtain the positions
    fixedPos = np.concatenate((np.expand_dims(xFixedPos, 0),
                                np.expand_dims(yFixedPos, 0)),
                                axis = 0)
    
    # Get rid of unnecessary agents
    fixedPos = fixedPos[:, 0:n_agents]
    
    # Adjust to correct shape
    fixedPos = fixedPos.T
    
    # Now generate the noise
    perturbPos = np.random.uniform(low = -distPerturb,
                                    high = distPerturb,
                                    size = (n_agents,  2))
    # Initial positions
    initPos = fixedPos + perturbPos
    
    if doPrint:
        print("OK", flush = True)
            
    return initPos

def compute_goals_initial_positions(X_0):
    """ 
    Generates a NumPy array with the 
    initial x, y position for each of the n_goals
    
    Parameters
    ----------
    X_0 : np.array (n_agents x 2) 
        Initial positions of the agents for all samples
    min_dist : double (legacy)
        The minimum distance between each agent
    
    Returns
    -------
    np.array (n_samples x n_goals x 2) 
    """

    n_goals = X_0.shape[0]

    goal_position = np.zeros((n_goals, 2))

    for goal in range(0, n_goals):
        x_0 = X_0[goal, 0]
        y_0 = X_0[goal, 1]
        radius = np.random.uniform(1, 1.5)
        phi = np.random.uniform(0, 2*np.math.pi)
        goal_position[goal] = np.array([radius * np.math.cos(phi) + x_0, radius * np.math.sin(phi) + y_0])
            
    return goal_position

def compute_communication_graph(X, degree):
        """ 
        Computes the communication graphs S.
        
        Parameters
        ----------
         X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        degree : int
            number of edges for each node (agent)
            
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x n_agents)
        
        """
        
        neigh = NearestNeighbors(n_neighbors=degree)
        neigh.fit(X)
        graphMatrix = np.array(neigh.kneighbors_graph(mode='connectivity').todense())    
            
        return graphMatrix
    
    
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom        
    
def select_action(policy, x, S, sigma):
    with torch.no_grad():
        action = policy(x, S)
        action = torch.t(action[0, -1])        
        
    noise = torch.normal(0, sigma, action.size()) # Generate noise from normal distribution
    prob = normpdf(torch.sum(noise), 0, sigma * action.numel()) # Get probability
    prob_sq = prob**2 # Square probability
    log_prob = math.log(prob_sq) # Take the log of the probability
    
    action = action + noise # Add noise to action

    return action.numpy(), log_prob

def update_policy(optimizer, gamma, reward_episode, policy_prob_history):
    t_samples = reward_episode.shape[0] 
    
    for t in range(0, t_samples):
        R = 0
        rewards = deque([])
        prob_history = torch.Tensor(policy_prob_history[t:-1])
        
        # Discount future rewards back to the present using gamma
        for r in reward_episode[t:-1]:
            R = r + gamma * R
            rewards.appendleft(R)
            
        # Scale rewards
        rewards = torch.FloatTensor(rewards)
                    
        # Calculate loss
        actions_reward_product = -torch.mul(prob_history, Variable(rewards)) # reward of each action
        loss = Variable(torch.sum(actions_reward_product, -1), requires_grad=True)

        # Update network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def save_model(thisFilename, archit):
    thisFilename = 'constrainedRL' # This is the general name of all related files
    saveDirRoot = 'experiments' # In this case, relative location
    saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the results from each run
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    saveModelDir = os.path.join(saveDir,'savedModels')
    if not os.path.exists(saveDir):
        os.makedirs(saveModelDir)
    saveFile = os.path.join(saveModelDir, 'localGNN')
    torch.save(archit.state_dict(), saveFile+'Archit'+ 'Last'+'.ckpt')
        
    
    
    
    