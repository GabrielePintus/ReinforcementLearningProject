import numpy as np
import gym
from tqdm.auto import tqdm
from Environment.market_making import MarketMakerEnv
from Environment.market_making import PhiTransform
from collections import defaultdict
from tqdm.auto import tqdm
from torch import nn
import torch

class QLearningAgent:
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.value_function = value_function  # Generalized value function approximator
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    # epsilon-greedy policy
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            q_values = self.value_function.get_q_values(state)
            return np.argmax(q_values)  # Exploit
    
    # Q-learning update
    def learn(self, state, action, reward, next_state, done):
        q_values = self.value_function.get_q_values(state)
        next_q_values = self.value_function.get_q_values(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(next_q_values)
        td_target = reward + (self.gamma * next_q_values[best_next_action] if not done else reward)
        td_error = td_target - q_values[action]
        
        self.value_function.update(state, action, self.alpha * td_error)
    
    # Decay epsilon - i.e. exploration rate
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Train the agent
    def train(self, n_episodes):
        progress_bar = tqdm(range(n_episodes), desc='Training', unit='episode')
        
        for episode in progress_bar:
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            self.update_epsilon()
            progress_bar.set_postfix({'Total reward': total_reward, 'Epsilon': self.epsilon})

    # Test the agent
    def test(self, n_episodes):
        all_rewards = []
        for episode in tqdm(range(n_episodes), desc='Testing', unit='episode'):
            state = self.env.reset()
            done = False
            rewards = []
            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
            rewards = np.array(rewards)
            all_rewards.append(rewards)
        return np.array(all_rewards).mean(axis=0)



class TileCoder:
    def __init__(self, num_tiles, num_tilings, state_bounds):
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.state_bounds = state_bounds
        # Consider different bounds for each dimension - TODO
        self.tile_widths = (state_bounds[1] - state_bounds[0]) / num_tiles
        self.tile_coders = [self._create_tile_coder() for _ in range(num_tilings)]

    def _create_tile_coder(self):
        def coder(state, tiling_index):
            scaled_state = (state - self.state_bounds[0]) / self.tile_widths
            tile_indices = np.floor(scaled_state + tiling_index).astype(int)
            return tuple(tile_indices)
        return coder

    def get_tiles(self, state):
        tiles = set()
        for tiling_index in range(self.num_tilings):
            tiles.add(self.tile_coders[tiling_index](state, tiling_index))
        return tiles

class TileCodingValueFunction:
    def __init__(self, num_tiles, num_tilings, state_bounds, action_space_n):
        self.tile_coder = TileCoder(num_tiles, num_tilings, state_bounds)
        self.Q = defaultdict(lambda: np.zeros(action_space_n))
    
    def get_q_values(self, state):
        tiles = self.tile_coder.get_tiles(state)
        q_values = np.mean([self.Q[tile] for tile in tiles], axis=0)
        return q_values
    
    def update(self, state, action, td_error):
        tiles = self.tile_coder.get_tiles(state)
        for tile in tiles:
            self.Q[tile][action] += td_error


# Linear Function Approximator
class LinearFunctionApproximator:
    def __init__(self, state_dim, action_dim):
        self.weights = np.zeros((action_dim, state_dim))
    
    def get_features(self, state, action):
        features = np.zeros(self.weights.shape[1])
        features[state] = 1  # Example: Simple encoding, adjust based on your state representation
        return features
    
    def get_q_values(self, state):
        q_values = np.dot(self.weights, state)  # Linear combination of features and weights
        return q_values
    
    def update(self, state, action, td_error, alpha):
        features = self.get_features(state, action)
        self.weights[action] += alpha * td_error * features  # Update rule for linear function approximator



# class QNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(QNetwork, self).__init__()
#         self.fcnn = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim)
#         )
        
#     def forward(self, state):
#         return self.fcnn(state)



# class ValueFunctionApproximator:
#     def __init__(self, model, optimizer):
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = nn.MSELoss()
        
#     def predict(self, state):
#         return self.model(state)
    
#     def update(self, state, target):
#         self.optimizer.zero_grad()
#         loss = self.loss_fn(self.model(state), target)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()