import numpy as np



pusher_actionspace_n = 7
pusher_actionspace_bounds = np.array([[-2, 2] for _ in range(pusher_actionspace_n)])


pusher_envspace_bounds = [
    [-3*np.pi, 3*np.pi] for _ in range(7)
]
pusher_envspace_bounds.extend([
    [-10, 10] for _ in range(7)
])
pusher_envspace_bounds.extend([
    [-10, 10] for _ in range(9)
])
pusher_envspace_bounds = np.array(pusher_envspace_bounds)

sa_bounds = np.concatenate([pusher_envspace_bounds, pusher_actionspace_bounds], axis=0)


