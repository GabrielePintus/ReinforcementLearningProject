import gym
import gym.spaces





class MarketMakerEnv(gym.Env):

    def __init__(self, lob_data):
        self.lob_depth = lob_data.shape[1]

        self.observation_space = gym.spaces.Dict({
            'order_book':       gym.spaces.Box( low = 0             , high=float("inf"), shape=(self.lob_depth, 4)   , dtype=float ),
            'market_features':  gym.spaces.Box( low = 0             , high=float("inf"), shape=(1,)     , dtype=float ),
            'inventory':        gym.spaces.Box( low = -float("inf") , high=float("inf"), shape=(1,)     , dtype=float )
        })

        self.action_space = gym.spaces.Dict({
            'order_size': gym.spaces.Box(low=1, high=float("inf"), shape=(1,), dtype=int),
            'order_side': gym.spaces.Discrete(2), # 0: buy, 1: sell
            'action_type': gym.spaces.Discrete(2) # 0: place order, 1: cancel order
        })

    def reset(self):
        self.t = 0

    def step(self, action):
        environment_obs = self.lob_data[self.t]

        # Check if the action is valid
        assert self.action_space.contains(action)

        # Execute the action
        if action['action_type'] == 0:
            # Place order
        else:
            # Cancel order


        # Update the state
        # Calculate the reward
        # Check if the episode is done
        # Return the state, reward, done, and info


    def render(self, mode='human'):
        pass

    def close(self):
        pass

