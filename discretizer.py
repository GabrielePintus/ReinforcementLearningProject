import numpy as np
import math


class TilingDiscretizer:

    def __init__(
        self,
        bounds : np.array,  # The domain bounds for each dimension
        n_tiles : int,      # The number of tiles in each dimension
        n_tilings : int,    # The number of tilings
        shifts : np.array,  # The shifts for each tiling
        single_idx : bool = False
    ):
        self.bounds = bounds
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.shifts = shifts
        self.single_idx = single_idx


    def rescale(self, x):
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def encode(self, x):
        # if x is a number convert it to a numpy array
        if not isinstance(x, (list, np.ndarray)):
            x = np.array([x])

        # Prepare the output
        if self.single_idx:
            z = np.zeros((self.n_tilings,))
        else:
            z = np.zeros((self.n_tilings, len(x)))

        # Rescale the input to [0, 1]
        x = self.rescale(x)

        for i in range(self.n_tilings):
            # Shift the input
            y = x + self.shifts * i
            # Bound the input
            # y = np.clip(y, 0, 1)
            y = (y % 1).round(8)
            # Discretize the input
            y = (y * self.n_tiles).astype(int)
            if self.single_idx:
                y = np.ravel_multi_index(y, [self.n_tiles] * len(y), order='C')
            z[i] = y
    
        return z.astype(int)


class TilingEncoder:

    def __init__(self, bounds, tiles, single_idx=False):
        self.bounds = bounds
        self.tiles = tiles
        self.single_idx = single_idx

    def rescale(self, x):
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def encode(self, x):
        # if x is a number convert it to a numpy array
        if not isinstance(x, (list, np.ndarray)):
            x = np.array([x])

        # Rescale
        y = self.rescale(x)

        # Bound
        # y = np.clip(y, 0, 1)
        y = (y % 1).round(8)

        # Discretize
        z = (y * self.tiles).astype(int)

        if self.single_idx:
            z = np.ravel_multi_index(z, [self.tiles] * len(z), order='C')

        return z

    def __call__(self, x):
        return self.encode(x)

        




if __name__ == '__main__':
    bounds = np.array([[0,3], [0,2]])
    x = np.array([.79, .81])
    n_tiles = 13

    tiling = TilingEncoder(bounds, n_tiles, False)
    print(tiling.encode(x))

    tiling = TilingEncoder(bounds, n_tiles, True)
    print(tiling.encode(x))


