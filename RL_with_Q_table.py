# use Python 3.8.3
import gym
import numpy as np
import math
from numpy import save

class CartPole():
    
    def __init__(self, buckets=(1, 1, 6, 14), n_episodes = 500, min_alpha=0.01, alpha = 1, adaptive_lr = 0.99, min_epsilon=0.1, gamma=1.0, epsilon = 1, exploration_decay=0.99):
        self.buckets = buckets # down-scaling feature space to discrete range (a matrix of 6 X 12)
        self.n_episodes = n_episodes # training episodes 
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.exploration_decay = exploration_decay # only for development purposes
        self.epsilon = epsilon # Exploration rate
        self.alpha = alpha
        self.adaptive_lr = adaptive_lr # Learning rate
        self.env = gym.make('CartPole-v1')
       
        # initialising Q-table
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    # Discretizing input space to make Q-table and to reduce dimmensionality
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))] # Get index from Q table
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))] # Get index from Q table
        return tuple(new_obs)

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation:
    # ******** New Q(s,a) = Q(s,a) + alpha [ (R(s,a)) + gama(max(Q'(s',a'))) - Q(s,a) ] **********
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive exploration rate
    def get_epsilon(self, e):
        # epsilon_decay
        self.epsilon *= self.exploration_decay
        return max(self.min_epsilon, self.epsilon)

    # Adaptive learning rate
    def get_alpha(self, t):
        self.alpha *= self.adaptive_lr
        return max(self.min_alpha, self.alpha)

    def run(self):
        for e in range(self.n_episodes):                    
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())

            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0
            rw = []
            while not done:
                # Render environment
                # self.env.render()
                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1
                rw.append(reward)
                if i >= 10000:
                    return
                
            print("Episode:", e+1,"  Acumulated reward:" ,sum(rw))   
        print(self.Q)
        save('Q.npy', self.Q)

if __name__ == "__main__":  
    # Make an instance of CartPole class 
    solver = CartPole()
    solver.run()
