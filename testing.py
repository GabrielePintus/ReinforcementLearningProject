import gym
import gym.spaces
import numpy as np

from market_making import MarketMakerEnv


# Order book test

# Crea uno spazio discreto di 2*4 dim tra 0 e 1e6, dovrebbe andare bene per Order Book

num_dimensions = 2 * 4
high = 1e6

limits = np.full([2,4], high)

test = gym.spaces.MultiDiscrete(limits)

print("Order book test: ", test.sample())



# Order size test

test_2 = gym.spaces.MultiDiscrete([MarketMakerEnv.MAX_ORDER_SIZE, MarketMakerEnv.MAX_ORDER_SIZE])

print("Order size test: ", test_2.sample())

# Theta test

test_3 = gym.spaces.Discrete(9)

print("Theta values test: ", test_3.sample())
