import gym
import gym.spaces
import numpy as np




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
    ])
    
    # Useful static methods
    @staticmethod
    def p(mid_price, theta, spread):
        return mid_price + theta * spread

    def __init__(self, lob_data: np.array, feature_extractor: callable):
        # Environment state = agent_state + market_state
        
        # Agent state = (inventory, spread, theta_a, theta_b)
        # 1. Inventory - the amount of stock currently owned or owed by the agent
        # 2. Spread - the current value of the spread scale factor
        # 3. Active quoting distances normalized by the current spread
        #    there are the effective values of the control parameters theta_a and theta_b
        #    after stepping forward in the simulation
        self.agent_state = np.zeros(4)
        
        # Market state = (ba_spread, mid_move, book_imbalance, signed_volume, volatility, rsi)
        # 1. Bid-Ask spread
        # 2. Mid price move
        # 3. Book imbalance
        # 4. Signed volume
        # 5. Volatility
        # 6. Relative Strength Index (RSI)
        self.market_state = np.zeros(6)
        
        # Action space
        # There are a total of 9 possible actions (buy, sell) that the agent can take
        # 1 = (1,1), ..., 5 = (5,5), 6 = (1,3), 7 = (3,1), 8 = (2,5), 9 = (5,2)
        self.action_space = gym.spaces.Discrete(9)
        
        # Additional variables
        self.reference_price = 0
        self.ask_book = dict()
        self.bid_book = dict()
        self.p_a = 0
        self.p_b = 0
    
    # Given the current open position, check if some of the orders can be matched and execute them
    def match(self):
        # check if there are any orders in the bid book
        # that can be matched with the ask book
        # Additionally, save the amount of volume matched
        matched_a = 0
        matched_b = 0
        for price in self.bid_book:
            if price in self.ask_book:
                if self.bid_book[price] > self.ask_book[price]:
                    matched_a += self.ask_book[price]
                    matched_b += self.ask_book[price]
                    self.bid_book[price] -= self.ask_book[price]
                    self.ask_book.pop(price)
                else:
                    matched_a += self.bid_book[price]
                    matched_b += self.bid_book[price]
                    self.ask_book[price] -= self.bid_book[price]
                    self.bid_book.pop(price)
                    
        return matched_a, matched_b
        
        
    

    def reset(self):
        self.agent_state = np.zeros_like(self.agent_state)
        self.market_state = np.zeros_like(self.market_state)
    
    def reward(self):
        """
            Simple Profit and Loss (PnL) reward function
        """        
        d_a = self.agent_state[2] * self.agent_state[1]
        d_b = -1 * self.agent_state[3] * self.agent_state[1]
        matched_a, matched_b = self.match()
        psi_a = matched_a * d_a
        psi_b = matched_b * d_b
        
        phi = self.agent_state[0] * self.market_state[1]
        Psi = psi_a + psi_b + phi
        
        return Psi

    def transition(self, action):
        """
            Transition function
            The action specifies which orders to place in the market both on the bid and ask side
            it is essentialy the theta values for the bid and ask side
        """
        theta_a, theta_b = self.ACTIONS_0_8[action]
        self.agent_state[2] = theta_a
        self.agent_state[3] = theta_b        
        

    def step(self, action):
        self.transition(action)
        reward = self.reward()
        done = False
        return self.agent_state, reward, done, {}

