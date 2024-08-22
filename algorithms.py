import numpy as np
import gym
from tqdm.auto import tqdm
from Environment.market_making import MarketMakerEnv
from Environment.market_making import PhiTransform
from collections import defaultdict
from tqdm.auto import tqdm

class TileCodingQLearningAgent:
    def __init__(self, env, num_tiles=10, num_tilings=8, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.tile_coder = TileCoder(num_tiles, num_tilings, env.observation_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # epsilon-greedy policy
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            tiles = self.tile_coder.get_tiles(state)
            q_values = np.mean([self.Q[tile] for tile in tiles], axis=0)
            return np.argmax(q_values)  # Exploit
    
    # Q-learning update
    def learn(self, state, action, reward, next_state, done):
        tiles = self.tile_coder.get_tiles(state)
        next_tiles = self.tile_coder.get_tiles(next_state)
        q_values = np.mean([self.Q[tile] for tile in tiles], axis=0)
        next_q_values = np.mean([self.Q[tile] for tile in next_tiles], axis=0)
        
        best_next_action = np.argmax(next_q_values)
        td_target = reward + (self.gamma * next_q_values[best_next_action] if not done else reward)
        td_error = td_target - q_values[action]
        
        for tile in tiles:
            self.Q[tile][action] += self.alpha * td_error
    
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
            progress_bar.update()
        
            

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
