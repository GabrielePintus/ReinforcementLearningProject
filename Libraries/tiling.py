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
        assert len(bounds) == state_dim
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
        
#         # Initialize the coefficient matrix with shape (n_tilings, n_tiles ** state_dim)
#         self.coefficients = np.zeros((n_tilings, n_tiles ** state_dim))
    
#     def get_feature_vector(self, encoding):
#         """
#         Create a sparse one-hot encoded feature vector for the given encoding.
#         """
#         feature_vector = np.zeros(self.coefficients.size)
#         for i in range(self.n_tilings):
#             tile_index = np.ravel_multi_index(encoding[i].astype(int), (self.n_tiles,) * self.state_dim)
#             feature_index = i * (self.n_tiles ** self.state_dim) + tile_index
#             feature_vector[feature_index] = 1  # Set the feature corresponding to the active tile to 1
#         return feature_vector
    
#     def evaluate(self, state):
#         """
#         Estimate the value of the given state using the dot product of the feature vector and coefficients.
#         """
#         encoding = self.encode(state)
#         feature_vector = self.get_feature_vector(encoding)
        
#         # Flatten the coefficients and compute the dot product
#         value = np.dot(self.coefficients.flatten(), feature_vector)
#         return value
    
#     def update(self, state, target, alpha=0.1):
#         """
#         Update the coefficients based on the target value and learning rate alpha.
#         """
#         encoding = self.encode(state)
#         feature_vector = self.get_feature_vector(encoding)
        
#         # Compute the current prediction
#         prediction = self.evaluate(state)
        
#         # Compute the error
#         error = target - prediction
        
#         # Update the coefficients for the active features
#         self.coefficients += (alpha * error * feature_vector).reshape(self.coefficients.shape)


# class LinearCombinationTileEncodingLight(TileEncodingApproximator):
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
#         self.coefficients = 



# from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
# class LinearCombinationTileEncodingSparse(TileEncodingApproximator):
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
        
#         # Initialize the coefficient matrix with shape (n_tilings, n_tiles ** state_dim)
#         # self.coefficients = np.zeros((n_tilings, n_tiles ** state_dim))
#         print(f'Creating sparse matrix with shape ({n_tilings}, {n_tiles ** state_dim})')
#         self.coefficients = csr_matrix((n_tilings, n_tiles ** state_dim), dtype=np.float32)
    
#     def get_feature_vector(self, encoding):
#         """
#         Create a sparse one-hot encoded feature vector for the given encoding.
#         """
#         feature_size = self.n_tilings * self.n_tiles ** self.state_dim
#         feature_vector = np.zeros(feature_size)
#         for i in range(self.n_tilings):
#             tile_index = np.ravel_multi_index(encoding[i].astype(int), (self.n_tiles,) * self.state_dim)
#             feature_index = i * (self.n_tiles ** self.state_dim) + tile_index
#             feature_vector[feature_index] = 1  # Set the feature corresponding to the active tile to 1
#         return feature_vector
    
#     def evaluate(self, state):
#         """
#         Estimate the value of the given state using the dot product of the feature vector and coefficients.
#         """
#         encoding = self.encode(state)
#         feature_vector = self.get_feature_vector(encoding)
        
#         # Flatten the coefficients and compute the dot product
#         # value = np.dot(self.coefficients.flatten(), feature_vector)
#         value = 0.0
#         for i in range(self.coefficients.shape[0]):
#             i_a = i*self.n_tiles**self.state_dim
#             i_b = (i+1)*self.n_tiles**self.state_dim            
#             value += self.coefficients[i,:].dot(feature_vector[i_a:i_b])
#         return value
    
#     def update(self, state, target, alpha=0.1):
#         """
#         Update the coefficients based on the target value and learning rate alpha.
#         """
#         encoding = self.encode(state)
#         feature_vector = self.get_feature_vector(encoding)
        
#         # Compute the current prediction
#         prediction = self.evaluate(state)
        
#         # Compute the error
#         error = target - prediction
        
#         # Update the coefficients for the active features
#         self.coefficients += (alpha * error * feature_vector).reshape(self.coefficients.shape)


#     def get_q_values(self, state):
#         return self.evaluate(state)



class LinearCombinationTileEncoding(TileEncodingApproximator):
    """
    Linear combination of tile encoding features for function approximation
    using dot product.
    """
    def __init__(
        self,
        state_dim: int,
        bounds: np.ndarray,
        n_tiles: int,
        n_tilings: int,
        offset: float
    ):
        super().__init__(state_dim, bounds, n_tiles, n_tilings, offset)
        
        # Initialize the coefficient matrix
        # The coefficient matrix shape will be (n_tilings, n_tiles^state_dim)
        # Each tile has a coefficient, and we have n_tilings such sets of coefficients
        self.coefficients = np.random.randn(self.n_tilings, *(self.n_tiles,) * self.state_dim)
    
    def evaluate(self, state):
        # Encode the state into tile indices
        encoded_tiles = self.encode(state)
        
        # The approximation is a linear combination of the active tiles' coefficients
        approximation = 0
        for i in range(self.n_tilings):
            indices = tuple(encoded_tiles[i].astype(int))
            approximation += self.coefficients[i][indices]
        
        return approximation
    
    def update(self, state, target, learning_rate):
        # Encode the state into tile indices
        encoded_tiles = self.encode(state)
        
        # Calculate the prediction for the given state
        prediction = self.evaluate(state)
        
        # Calculate the error (difference between target and prediction)
        error = target - prediction
        
        # Update each of the coefficients associated with the active tiles
        for i in range(self.n_tilings):
            indices = tuple(encoded_tiles[i].astype(int))
            self.coefficients[i][indices] += learning_rate * error
            
    def get_q_values(self, state):
        return self.evaluate(state)
    
    
    
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
        for i in range(self.n_tilings):
            index = self._hash_indices(encoded_tiles[i], i)
            approximation += self.weights[index]
        return approximation
    
    def update(self, state, target, alpha):
        encoded_tiles = self.encode(state)
        prediction = self.evaluate(state)
        error = target - prediction
        
        for i in range(self.n_tilings):
            index = self._hash_indices(encoded_tiles[i], i)
            self.weights[index] += alpha * error
    
    def _hash_indices(self, tile_indices, tiling_index):
        """
        Hash the tile indices and tiling index into a single index for the weight vector.
        This uses a simple hash function, but can be made more sophisticated if needed.
        """
        return int((np.sum(tile_indices) + tiling_index) % self.n_weights)

    def get_q_values(self, state):
        return self.evaluate(state)


if __name__ == '__main__':
    import psutil
    import os
    
    def memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        megabytes = mem_info.rss / 1024 / 1024
        return megabytes
    
    
    
    # Example usage
    state_dim = 2
    bounds = np.array([[0, 1], [0, 1]])
    n_tiles = 8
    n_tilings = 4
    offset = 0.2
    
    state = np.array([0.5, 0.75])
    target_value = 2.3

    all_values = []
    for n_tilings in [1, 2, 4, 8]:
        value_function = SparseTileEncodingApproximator(state_dim, bounds, n_tiles, n_tilings, offset, n_weights=10)
        print(f'RAM memory usage: {memory_usage()} MB')

        
        value = value_function.evaluate(state)
        # print("Initial value:", value)

        # Update the value function
        values = []
        for i in range(50):
            value_function.update(state, target=target_value, alpha=0.05)
            value = value_function.evaluate(state)
            values.append(value)
        all_values.append(values)        
    
    import matplotlib.pyplot as plt
    
    for i, values in enumerate(all_values):
        plt.plot(values, label=f'{2**(i)} tilings')
    
    # Add horizontal line for target value
    plt.axhline(y=target_value, color='r', linestyle='--', label='Target value')
    plt.xlabel('Update steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()