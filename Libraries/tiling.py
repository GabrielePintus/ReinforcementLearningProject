import numpy as np
from enum import Enum



        
        
class TileEncodingApproximator:
    """
        Discritize the state space into tiles
    """
    def __init__(
        self,
        state_dim : int,
        bounds : np.ndarray,
        n_tiles: int,
        n_tilings : int,
        offset : float
    ):
        # Check the input arguments
        # assert len(bounds) == state_dim + 1
        assert n_tiles > 0
        assert n_tilings > 0
        assert offset >= 0 and offset < 1
        
        # Store the input arguments
        self.state_dim = state_dim
        self.bounds = bounds
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.offset = offset       
    
    @staticmethod
    def encode_vector(x, n, c):
        d = 1 + np.floor((x-c)) - np.ceil((x-c))
        f = np.floor((x-c) * n) + d * ((n - 1))
        g = f - n * np.floor((x-c))
        return g
    
    @staticmethod
    def normalize_state(state, bounds):
        state = np.array(state)
        return (state - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    def encode(self, state):
        # This should be parallelized
        normalized_state = TileEncodingApproximator.normalize_state(state, self.bounds)
        encoded = []        
        for i in range(self.n_tilings):
            encoding = TileEncodingApproximator.encode_vector(
                normalized_state, self.n_tiles, self.offset + i / self.n_tilings
            )
            encoded.append(encoding)
        return np.array(encoded)
    
    def evaluate(self, state):
        raise NotImplementedError('Subclass must implement abstract method')
    
        


# class LinearCombinationTileEncoding(TileEncodingApproximator):
#     """
#     Linear combination of tile encoding features for function approximation
#     using dot product.
#     """
#     def __init__(
#         self,
#         state_dim: int,
#         bounds: np.ndarray,
#         n_tiles: int,
#         n_tilings: int,
#         offset: float
#     ):
#         super().__init__(state_dim, bounds, n_tiles, n_tilings, offset)
        
#         # Initialize the coefficient matrix
#         # The coefficient matrix shape will be (n_tilings, n_tiles^state_dim)
#         # Each tile has a coefficient, and we have n_tilings such sets of coefficients
#         self.coefficients = np.random.randn(self.n_tilings, *(self.n_tiles,) * self.state_dim)
    
#     def evaluate(self, state):
#         # Encode the state into tile indices
#         encoded_tiles = self.encode(state)
        
#         # The approximation is a linear combination of the active tiles' coefficients
#         approximation = 0
#         for i in range(self.n_tilings):
#             indices = tuple(encoded_tiles[i].astype(int))
#             approximation += self.coefficients[i][indices]
        
#         return approximation
    
#     def update(self, state, target, learning_rate):
#         # Encode the state into tile indices
#         encoded_tiles = self.encode(state)
        
#         # Calculate the prediction for the given state
#         prediction = self.evaluate(state)
        
#         # Calculate the error (difference between target and prediction)
#         error = target - prediction
        
#         # Update each of the coefficients associated with the active tiles
#         for i in range(self.n_tilings):
#             indices = tuple(encoded_tiles[i].astype(int))
#             self.coefficients[i][indices] += learning_rate * error
            
#     def get_q_values(self, state):
#         return self.evaluate(state)
    
# class LinearCombinationTileEncoding(TileEncodingApproximator):



class SparseTileEncodingApproximator(TileEncodingApproximator):
    """
    Tile coding with hashing for a memory-efficient linear combination of features.
    """
    def __init__(
        self,
        state_dim: int,
        bounds: np.ndarray,
        n_tiles: int,
        n_tilings: int,
        offset: float,
        n_weights: int
    ):
        super().__init__(state_dim, bounds, n_tiles, n_tilings, offset)
        
        # Initialize a sparse weight vector with a fixed size
        self.weights = np.zeros(n_weights)
        self.n_weights = n_weights
    
    def evaluate(self, state):
        encoded_tiles = self.encode(state)
        approximation = 0
        indexes = self._hash_indices(encoded_tiles, np.arange(self.n_tilings))
        approximation = np.sum(self.weights[indexes])
        return approximation
    
    def update(self, state, target, alpha):
        encoded_tiles = self.encode(state)
        prediction = self.evaluate(state)
        error = target - prediction
        
        indexes = self._hash_indices(encoded_tiles, np.arange(self.n_tilings))
        indexes = np.array(indexes)
        alpha_error = alpha * error
        self.weights[indexes] += alpha_error
    
    def _hash_indices(self, tile_indices, tiling_index):
        """
        Hash the tile indices and tiling index into a single index for the weight vector.
        This uses a simple hash function, but can be made more sophisticated if needed.
        """
        return ((np.sum(tile_indices) + tiling_index) % self.n_weights).astype(int)

    def get_q_values(self, state, actions):
        q_values = dict()
        for action in actions:
            state_couple = np.concatenate((state, [action]))
            q_values[action] = self.evaluate(state_couple)
        return q_values





class LCTC_ValueFunction:
    """
        Linear Combination of Tile Coding Value Function
        You can use this class to combine multiple value functions
    """

    def __init__(self, value_functions, lctc_weights):
        self.value_functions = value_functions
        self.lctc_weights = lctc_weights
        self.n = len(value_functions)

    def evaluate(self, states):
        values = np.zeros(self.n)
        for i, state in enumerate(states):
            values[i] = self.value_functions[i].evaluate(state)
        value = np.dot(values, self.lctc_weights)
        return value
    
    def update(self, states, target, alpha):
        for i, state in enumerate(states):
            self.value_functions[i].update(state, target, alpha)

    def get_q_values(self, states, actions):
        q_values = dict()
        for action in actions:
            state_couples = [np.concatenate((state, [action])) for state in states]
            q_values[action] = self.evaluate(state_couples)
        return q_values


        





if __name__ == '__main__':
    import psutil
    import os
    
    def memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        megabytes = mem_info.rss / 1024 / 1024
        return megabytes
    
    
    
    # Example usage
    bounds_1 = np.array([[0, 1], [0, 1], [0,2]])
    bounds_2 = np.array([[0, 1], [0, 1], [0, 1], [0,2]])
    n_tiles = 8
    n_tilings = 4
    offset = 0.2
    
    state_1 = np.array([0.5, 0.75])
    state_2 = np.array([0.1, 0.9, 0.2])
    state_couple_1 = np.concatenate((state_1, [1]))
    state_couple_2 = np.concatenate((state_2, [2]))
    target_value = 2.3

    value_function_1 = SparseTileEncodingApproximator(state_couple_1.size, bounds_1, n_tiles, n_tilings, offset, 100)
    value_function_2 = SparseTileEncodingApproximator(state_couple_2.size, bounds_2, n_tiles, n_tilings, offset, 100)

    # Value function 1,2 evaluation
    evaluation = value_function_1.evaluate(state_couple_1)
    print('Initial evaluation 1:', evaluation)
    evaluation = value_function_2.evaluate(state_couple_2)
    print('Initial evaluation 2:', evaluation)

    value_functions = [value_function_1, value_function_2]
    lctc_weights = np.array([0.5, 0.5])
    value_function = LCTC_ValueFunction(value_functions, lctc_weights)

    print('Memory usage: {:.2f} MB'.format(memory_usage()))

    evaluation = value_function.evaluate([state_couple_1, state_couple_2])
    print('Initial evaluation:', evaluation)
    # Update the value function
    value_function.update([state_couple_1, state_couple_2], target_value, 0.1)
    evaluation = value_function.evaluate([state_couple_1, state_couple_2])
    print('Updated evaluation:', evaluation)

    # Get the Q-values
    actions = [0, 1, 2]
    q_values = value_function.get_q_values([state_1, state_2], actions)
    # print the Q-values
    for action, q_value in q_values.items():
        print('Q-value for action {}: {:.2f}'.format(action, q_value))


