import gym
import gym.spaces
import numpy as np





class MarketMakerEnv(gym.Env):

    MAX_ORDER_SIZE = 1000
    MAX_INVENTORY  = 10000.0

    def __init__(self, lob_data: np.array, feature_extractor: callable):
        self.lob_depth = lob_data.shape[1]
        self.lob_data = lob_data
        self.feature_extractor = feature_extractor
        # Agent's internal state
        self.inventory = 0

        self.observation_space = gym.spaces.Dict({
            'order_book':       gym.spaces.Box( low = 0 , high=float("inf"), shape=(self.lob_depth*2, 4)   , dtype=float ),
        })

        self.action_space = gym.spaces.Dict({
            # 0: Ask, 1: Bid
            'order_size': gym.spaces.Box(low=0, high=MarketMakerEnv.MAX_ORDER_SIZE, shape=(2,), dtype=int),
        })

    def reset(self):
        self.t = 0
        self.inventory = 0
        return self._get_obs()



    def _get_obs(self):
        lob_obs = self.lob_data[self.t,:]
        features_obs = self.feature_extractor(lob_obs)
        return {
            'order_book': lob_obs,
            'features': features_obs,
            'inventory': np.array([self.inventory], dtype=float)
        }

    # Simple PnL
    def reward(self, prev_state, action, next_state):
        reward = 0

        # Change in the mid price
        prev_mid_price = prev_state['features'][0]
        next_mid_price = next_state['features'][0]        
        
        # Change in the value of the inventory
        next_inventory = next_state['inventory']
        diff_inventory = ( next_mid_price - prev_mid_price ) * next_inventory
        reward += diff_inventory

        # Profit derived from order execution
        order_size = action['order_size']
        p_a = next_state['order_book'][0]
        p_b = next_state['order_book'][2]
        psi_a = order_size[0] * ( p_a - next_mid_price )
        psi_b = order_size[1] * ( next_mid_price - p_b )
        reward += psi_a + psi_b

        return reward



    # Given a state and an action, return the next state
    def transition(self, current_state, action):
        next_state = current_state.copy()
        order_size = action['order_size']
        order_side = action['order_side']

        # Place order
        if order_side == 0:
            next_state['order_book'][self.t,1] += order_size[0]
        else:
            next_state['order_book'][self.t,3] += order_size[1]

        # Update time
        self.t += 1
        
        return next_state
    

    def step(self, action):
        # Get action from agent
        order_size = action['order_size']   
        order_side = action['order_side']
        action_type = action['action_type']

        # Compute market features
        current_state = self._get_obs()

        # Perform action
        next_state = self.transition(current_state, action)

        # Compute reward
        reward = self.reward(current_state, action, next_state)

        # Check if episode is over
        done = False
        if self.t == self.lob_data.shape[0]:
            done = True

        return next_state, reward, done, {}
        





        
        


    def render(self, mode='human'):
        pass

    def close(self):
        pass

