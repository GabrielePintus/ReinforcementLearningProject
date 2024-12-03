import numpy as np
  

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

    def rescale(self, x):
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def encode(self, x):
        """
        Encode a continuous value into a discrete value.
        """
        assert type(x) == np.ndarray, 'x must be a numpy array.'

        # Rescale
        y = self.rescale(x)

        # Bound
        # y = np.clip(y, 0, 1)
        y = (y % 1).round(8)

        # Discretize
        z = (y * self.n_tiles).astype(int)

        if self.single_idx:
            z = np.ravel_multi_index(z, [self.n_tiles] * len(z), order='C')

        return np.array(z)
        
    def decode(self, z):
        """
        Decode a discrete value into a continuous value.
        """
        assert type(z) == np.ndarray, 'z must be a numpy array.'

        if self.single_idx:
            z = np.unravel_index(z, [self.n_tiles] * len(self.bounds))
            z = np.array(z)
        z = z / self.n_tiles
        z = z * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return np.array(z)



if __name__ == '__main__':
    bounds = np.array([[0,3], [0,2]])
    x = np.array([.79, .81])
    n_tiles = 10

    tiling = GridSpace(bounds, n_tiles, True)
    encoding = tiling.encode(x)
    decoding = tiling.decode(encoding)

    print('x:', x)
    print('encoding:', encoding)
    print('decoding:', decoding)

    print("Domain", tiling.domain)
    print("Codomain", tiling.codomain)


