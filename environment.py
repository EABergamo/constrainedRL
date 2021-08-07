import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial.distance import cdist
import utils
from os import path
import configparser



class unlabelledPlanningEnvironment(gym.Env):
    def __init__(self) -> None:
        
        config_file = path.join(path.dirname(__file__), "unlabelled_planning.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['default']
        
        self.n_agents = config['n_agents']
        self.accel_max = config['accel_max']
        self.degree = config['degree']
        self.eta = config['eta']
        self.delta = config['delta']
        self.R = config['R']
        
        self.t_samples = 30

        # Features per node, corresponds to the position and velocity 
        # of the neighboors (2*degree and 2*degree, respectively), the
        # position of the closest goals (2 * degree), the node's
        # own agent and velocity (2 * 2) and the corresponding lambda.
        self.n_features = 2 * (3 * self.degree + 2) + 1
               
        # Agent position 
        self.X = np.zeros((self.n_agents, 2))
        self.X = utils.compute_agents_initial_positions(self.n_agents, 6)
        
        # Agent position
        self.V = np.zeros((self.n_agents, 2))
        
        # Goal position
        self.G = utils.compute_goals_initial_positions(self.X)

        # Adjacency graph 
        self.Graph = np.zeros((self.n_agents, 
                               self.n_agents))
        # State vector
        self.State = np.zeros((self.n_agents, self.n_features))
        
        # Initial conditions
        self.State[0, :, :], self.Graph[0, :, :], _ = self._get_observation()

    def step(self, action):
        """ 
        Responsible for computing one step in the current environment for a given action
        
        Parameters
        ----------
        action : np.array (n_agents x 2) 
                 acceleration for the current time instant

        Returns
        -------
        observation (np.array (n_agents, n_features)), cost (double), done (boolean)
        """
        
        # Position, velocity update
        self.V = self.V + action * 0.1
        self.X = self.X + self.V * 0.1 + action ** 0.1 / 2
        
        # Observations
        obs, curr_graph, min_dist = self._get_observation() # Observations
        self.State = obs
        self.Graph = curr_graph
        
        observation = [obs, curr_graph]
        
        # Reward
        reward, done = self._get_reward(min_dist)
        
        return observation, reward, done, {}
    
    def _get_reward(self, min_dist):
        # Checks which goals have agents at distance at least R
        goals_completed = np.min(self.State[-self.degree * 2 - 1:-1, :], axis=0) < self.R
        reward = np.sum(goals_completed)
        
        done = reward == self.n_agents
        
        reward = reward + self.State[: , -1] * min_dist
        
        return reward, done
    
    def _get_observation(self):
        """ 
        Responsible for computing one step in the current environment for a given action
        
        Parameters
        ----------
        N/A

        Returns
        -------
        observation (np.array (n_agents, n_features))
        """
        degree = self.degree
        curr_state = np.zeros((self.n_agents, self.n_features))
        
        # Lambda update
        curr_graph = utils.compute_communication_graph(self.X, self.degree) # Communication graph
        distance_matrix = cdist(self.X, self.X) # Minimum distance between agents
        neighboorhood_distance = distance_matrix * curr_graph  # Position to neighboors only
        min_dist = np.min(neighboorhood_distance, axis=1) # Minimum distance to neighboors
        
        
        for agent in range(0, self.n_agents):
            # Own position, velocity
            curr_state[0:2, agent] = self.X[agent,:].flatten()
            curr_state[2:4, agent] = self.V[agent,:].flatten()
            
            # Other agents position, velocity
            closest_agents_index = curr_graph[agent, :] == 1
            curr_state[4:(degree+2)*2, agent] = (self.X[closest_agents_index] - np.tile(curr_state[0:2, agent], (self.degree, 1))).flatten()
            curr_state[(degree+2)*2:(2*degree + 2)*2, agent] = (self.V[closest_agents_index] - np.tile(curr_state[2:4, agent], (self.degree, 1))).flatten() 
        
            # Goals
            distance_matrix = cdist(self.X, self.G)
            distance_to_goals = distance_matrix[agent, :]
            closest_goals_index = np.argsort(distance_to_goals)[0:degree]
            curr_state[-degree * 2 - 1:-1, agent] = (self.G[closest_goals_index] - np.tile(curr_state[0:2, agent], (self.degree, 1))).flatten()
            
            # Lambda
            curr_state[agent, -1] = np.max(0, self.State[agent, -1] + self.eta / self.t_samples * (min_dist[agent] - self.delta))
            
        return curr_state, curr_graph, min_dist
    
    def reset(self):
        # Agent position 
        self.X = np.zeros((self.n_agents, 2))
        self.X = utils.compute_agents_initial_positions(self.n_agents, 6)
        
        # Agent position
        self.V = np.zeros((self.n_agents, 2))
        
        # Goal position
        self.G = utils.compute_goals_initial_positions(self.X)
        
        # Adjacency graph 
        self.Graph = np.zeros((self.n_agents, 
                               self.n_agents))
        # State vector
        self.State = np.zeros((self.n_agents, self.n_features))
        
        # Initial conditions
        self.State[0, :, :], self.Graph[0, :, :], _ = self._get_observation()
        
        observation = [self.State[0, :, :], self.Graph[0, :, :]]
        
        return observation
            
   