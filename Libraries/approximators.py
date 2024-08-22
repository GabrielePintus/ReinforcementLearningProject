from collections import defaultdict
import numpy as np


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
    
    def update(self, state, action, td_error, *args):
        tiles = self.tile_coder.get_tiles(state)
        for tile in tiles:
            self.Q[tile][action] += td_error


# Linear Function Approximator
class LinearFunctionApproximator:
    def __init__(self, state_dim, action_dim, alpha=0.1):
        self.weights = np.zeros((action_dim, state_dim))
        self.alpha = alpha
    
    def get_features(self, state, action):
        features = np.zeros(self.weights.shape[1])
        features[state] = 1  # Example: Simple encoding, adjust based on your state representation
        return features
    
    def get_q_values(self, state):
        q_values = np.dot(self.weights, state)  # Linear combination of features and weights
        return q_values
    
    def update(self, state, action, td_error):
        features = self.get_features(state, action)
        self.weights[action] += self.alpha * td_error * features  # Update rule for linear function approximator
