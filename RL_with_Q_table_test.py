# use Python 3.8.3
import gym
import numpy as np
import math
from numpy import load

env = gym.make('CartPole-v1').unwrapped
Q = np.load('Q_learning_model.npy', allow_pickle=True)
buckets = np.shape(Q)
bucket = np.delete(buckets, -1)
episodes = 1

# Discretizing input space to reduce dimmensionality
def discretize(obs):  
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)] # Getting upper bound
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)] # Getting lower bound
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))] 
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))] # Get index from Q table 
    return tuple(new_obs)

for e in range(episodes):                        
    current_state = discretize(env.reset())
    done = False
    i = 0
    while not done:
        env.render() # Render environment    
        action = np.argmax(Q[(current_state)]) # take action using the Q-table
        new_state, reward, done, info = env.step(action) # get new observations
        current_state = discretize(new_state) # discretise observations so Q-table can be use
        i += 1 
        print("Reward = ",i)    
    print("Episode:", e+1,"  Acumulated reward:" ,i)            
    