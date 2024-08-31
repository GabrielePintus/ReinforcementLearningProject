import gym
import gym.spaces
import numpy as np
from numpy import inf
from numpy.core.multiarray import array as array



class PhiTransform:
    
    @staticmethod
    def PnL():
        return lambda x: x
    
    @staticmethod
    def PnL_dampened(c=0.5):
        return lambda x: (1-c) * x
    
    @staticmethod
    def PnL_asymm_dampened(c=-0.5):
        return lambda x: x * (1-c) - np.abs(c*x)
        


class MarketMakerEnv(gym.Env):

    # Constants
    MAX_ORDER_SIZE = 1e3
    MAX_INVENTORY_SIZE  = 1e4
    ACTIONS_0_8 = np.array([
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        [5,5],
        [1,3],
        [3,1],
        [2,5],
        [5,2],
        [-1,-1],
    ])
    
    # Useful static methods
    @staticmethod
    def p(mid_price, theta, spread):
        # ref + dist
        return mid_price + theta * spread

    def __init__(
        self,
        lob_data: np.array,
        horizon=np.inf,
        phi_transform=lambda x: x,
    ):
        # Environment state = agent_state + observation_space
        
        # Agent state = (inventory, spread, theta_a, theta_b)
        # 1. Inventory - the amount of stock currently owned or owed by the agent
        # 2. Spread - the current value of the spread scale factor
        # 3. Active quoting distances normalized by the current spread
        #    there are the effective values of the control parameters theta_a and theta_b
        #    after stepping forward in the simulation
        self.agent_state = np.zeros(4)
        
        # Market state = (ba_spread, mid_move, book_imbalance, signed_volume, volatility, rsi)
        # 0. Mid price
        # 1. Bid-Ask spread
        # 2. Mid price move
        # 3. Book imbalance
        # 4. Signed volume
        # 5. Volatility
        # 6. Relative Strength Index (RSI)
        self.observation_space = np.zeros(6)
        
        # Action space
        # There are a total of 9 possible actions (buy, sell) that the agent can take
        # 1 = (1,1), ..., 5 = (5,5), 6 = (1,3), 7 = (3,1), 8 = (2,5), 9 = (5,2)
        self.action_space = gym.spaces.Discrete(10)
        
        
        # Additional variables
        self.t = 0
        self.horizon = horizon
        self.ask_book = dict()
        self.best_ask = 0
        self.bid_book = dict()
        self.best_bid = 0
        self.market_book_data = lob_data
        self.p_a = 0
        self.p_b = 0
        self.v_a = gym.spaces.Box(low=0, high=self.MAX_ORDER_SIZE, shape=(1,), dtype=np.int32)
        self.v_b = gym.spaces.Box(low=0, high=self.MAX_ORDER_SIZE, shape=(1,), dtype=np.int32)
        self.market_order_size = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.phi_transform = phi_transform

        # Variables for additional information
        self.total_orders_placed = 0
        self.total_orders_executed = 0

        # State space dimension
        self.states_dim = [len(s) for s in self.get_state()]
        self.state_dim = sum(self.states_dim)
    
    # Given the current open position, check if some of the orders can be matched and execute them
    def match(self):
        # check if there are any orders in the bid book
        # that can be matched with the ask book
        # Additionally, save the amount of volume matched
        matched_a = 0
        matched_b = 0
        market_book = self.market_book_data[self.t,:]

        # Check if i can buy from someone in the market
        for price, volume in self.ask_book.items():
            if price <= market_book[2]:
                v = min(volume, market_book[3])
                matched_a += v
                self.ask_book[price] -= v
                self.total_orders_executed += 1
        # Check if i can sell to someone in the market
        for price, volume in self.bid_book.items():
            if price >= market_book[0]:
                v = min(volume, market_book[1])
                matched_b += v
                self.bid_book[price] -= v
                self.total_orders_executed += 1
                

        return matched_a, matched_b
        
    def place_order(self, price, volume, side):
        if side == 'ask':
            self.ask_book[price] = self.ask_book.get(price, 0) + volume
            self.best_ask = min(self.ask_book.keys())
        elif side == 'bid':
            self.bid_book[price] = self.bid_book.get(price, 0) + volume
            self.best_bid = max(self.bid_book.keys())
        
        # Update the total number of orders placed
        self.total_orders_placed += 1
    

    def reset(self):
        self.t = 0
        self.agent_state = np.zeros_like(self.agent_state)
        self.observation_space = np.zeros_like(self.observation_space)
        self.observation_space = self.market_book_data[self.t,:]
        self.ask_book = dict()
        self.bid_book = dict()
        self.total_orders_placed = 0
        self.total_orders_executed = 0
        
        return self.get_state()
    
    def reward(self):
        """
            Simple Profit and Loss (PnL) reward function
        """        
        # Compute psi_a and psi_b
        d_a = self.agent_state[2] * self.agent_state[1]
        d_b = -1 * self.agent_state[3] * self.agent_state[1]
        matched_a, matched_b = self.match()
        psi_a = matched_a * d_a
        psi_b = matched_b * d_b
        
        # Compute phi
        phi = self.agent_state[0] * self.observation_space[6]
        phi = self.phi_transform(phi)
        
        # Compute the total reward
        Psi = psi_a + psi_b + phi
        
        return Psi, psi_a, psi_b, phi
    
    def clear_inventory(self):
        """
            Clear the inventory at the end of the episode
        """
        return self.agent_state[0] * self.observation_space[0]
    

    def transition(self, action):
        """
            Transition function
            The action specifies which orders to place in the market both on the bid and ask side
            it is essentialy the theta values for the bid and ask side
        """
        theta_a, theta_b = self.ACTIONS_0_8[action]
        if theta_a < 0 and theta_b < 0:
            # clear inventory 
            total_ask = sum(self.ask_book.values())
            total_bid = sum(self.bid_book.values())
            balance_volume = total_ask - total_bid
            # market_order_size = self.market_order_size.sample()[0] * balance_volume
            market_order_size = 1 * balance_volume
            if balance_volume > 0:
                self.place_order(self.best_bid, market_order_size, 'bid')
            elif balance_volume < 0:
                self.place_order(self.best_ask, -market_order_size, 'ask')
        else:
            self.agent_state[2] = theta_a
            self.agent_state[3] = theta_b
            # Place orders in the market
            # First compute the half spread
            spread = self.observation_space[5] / 2
            self.agent_state[1] = spread
            # Compute the bid and ask prices along with the volume
            p_a = self.p(self.observation_space[4], -theta_a, spread)
            p_b = self.p(self.observation_space[4], theta_b, spread)
            v_a = self.v_a.sample()[0]
            v_b = self.v_b.sample()[0]
            if self.agent_state[0] >= MarketMakerEnv.MAX_INVENTORY_SIZE:
                v_a = 0
            if self.agent_state[0] <= -MarketMakerEnv.MAX_INVENTORY_SIZE:
                v_b = 0
            # Update the bid and ask books
            self.place_order(p_a, v_a, 'ask')
            self.place_order(p_b, v_b, 'bid')

        # Execute the orders
        # Update the time
        self.t += 1
        matched_a, matched_b = self.match()
        # Update the inventory
        self.agent_state[0] += matched_a - matched_b


    def get_state(self):
        return [self.observation_space]

    def step(self, action):
        self.transition(action)
        reward, psi_a, psi_b, phi = self.reward()
        done = self.t >= self.horizon
        info = {
            'total_orders_placed': self.total_orders_placed,
            'total_orders_executed': self.total_orders_executed,
            'inventory': self.agent_state[0],
            'reward': reward,
            'PnL' : psi_a + psi_b + phi,
            'psi_a': psi_a,
            'psi_b': psi_b,
            'phi': phi,
        }
        # Non andiamo mai avanti nel LOB senza questo?
        self.observation_space = self.market_book_data[self.t,:]
        return self.get_state(), reward, done, info



class Case1MarketMakerEnv(MarketMakerEnv):

    def __init__(self, lob_data: np.array, horizon=np.inf, phi_transform=lambda x: x):
        super().__init__(lob_data, horizon, phi_transform)


    # Add other 'get_X_state' methods and override 'get_state' method
    def get_agent_state(self):
        return self.agent_state
    
    def get_state(self):
        return [
            self.get_agent_state(),
        ]

class Case2MarketMakerEnv(MarketMakerEnv):

    def __init__(self, lob_data: np.array, horizon=np.inf, phi_transform=lambda x: x):
        super().__init__(lob_data, horizon, phi_transform)


    # Add other 'get_X_state' methods and override 'get_state' method
    def get_agent_state(self):
        return self.agent_state
    
    def get_market_state(self):
        return self.observation_space
    
    def get_state(self):
        return [
            self.get_agent_state(),
            self.get_market_state()
        ]

class Case3MarketMakerEnv(MarketMakerEnv):

    def __init__(self, lob_data: np.array, horizon=np.inf, phi_transform=lambda x: x):
        super().__init__(lob_data, horizon, phi_transform)


    # Add other 'get_X_state' methods and override 'get_state' method
    def get_agent_state(self):
        return self.agent_state
    
    def get_market_state(self):
        return self.observation_space
    
    def get_full_state(self):
        # Concatenate the agent state and the market state
        full = np.concatenate([self.get_agent_state(), self.get_market_state()])
        return full
    
    def get_state(self):
        return [
            self.get_agent_state(),
            self.get_market_state(),
            self.get_full_state()
        ]