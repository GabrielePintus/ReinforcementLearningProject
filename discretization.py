import numpy as np
from functools import wraps
import gymnasium as gym



def input_to_numpy(func):
    """Decorator to ensure the input is converted to a NumPy array."""
    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        # Convert input to a NumPy array
        x = np.array(x)
        return func(self, x, *args, **kwargs)
    return wrapper
  

class GridSpace:

    """
    A class to encode and decode continuous values into discrete values.
    
    Parameters
    ----------
    bounds : np.ndarray
        The bounds of the continuous space.
    n_tiles : int
        The number of tiles to discretize the space.
    single_idx : bool
        Whether to use a single index to encode the values.
    """
    

    def __init__(self, bounds, n_tiles, single_idx=False):
        self.bounds = bounds
        self.n_tiles = n_tiles
        self.single_idx = single_idx

        self.domain = self.bounds
        self.codomain = np.array([[0, n_tiles]] * len(bounds))
        self.n_states = n_tiles ** len(bounds)

    def rescale(self, x):
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    @input_to_numpy
    def encode(self, x):
        """
        Encode a continuous value into a discrete value.
        """

        # Rescale
        y = self.rescale(x)

        # Bound
        y = np.clip(y, 0, 1-1e-10)

        # Discretize
        z = (y * self.n_tiles).astype(int)

        if self.single_idx:
            z = np.ravel_multi_index(z, [self.n_tiles] * len(z), order='C')

        return np.array(z)
    
    @input_to_numpy
    def decode(self, z):
        """
        Decode a discrete value into a continuous value.
        """

        if self.single_idx:
            z = np.unravel_index(z, [self.n_tiles] * len(self.bounds))
            z = np.array(z)
        z = z / self.n_tiles
        z = z * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return np.array(z)





class DiscretizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins):
        super().__init__(env)
        self.n_bins = n_bins
        assert len(n_bins) == env.observation_space.shape[0], "n_bins must match the observation dimensions."

        # Create discretization grids for each observation dimension
        low = env.observation_space.low
        high = env.observation_space.high
        self.discrete_bins = [
            np.linspace(low[i], high[i], n_bins) for i, n_bins in enumerate(n_bins)
        ]

        # Compute the total number of discrete states
        self.observation_space = gym.spaces.Discrete(np.prod(n_bins))

    def observation(self, observation):
        # Map continuous observations to discrete bins
        discrete_obs = [
            np.digitize(np.clip(observation[i], self.discrete_bins[i][0], self.discrete_bins[i][-1]), self.discrete_bins[i]) - 1 for i in range(len(self.n_bins))
        ]
        return np.ravel_multi_index(discrete_obs, self.n_bins)