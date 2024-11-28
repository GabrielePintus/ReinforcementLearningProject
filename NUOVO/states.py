import numpy as np
import pandas as pd
from collections import deque
from order import Order



class PsiDampening:

    @staticmethod
    def symm_damp(psi, eta):
        return psi * (1 - eta)

    @staticmethod
    def asymm_damp(psi, eta):
        return psi - max(0, eta*psi)



class EnvironmentState:

    # Constants
    ACTION_SPACE = (
        # Symmetric prices
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        # Asymmetric prices
        (1, 3),
        (3, 1),
        (2, 5),
        (5, 2),
        # Clear inventory
        (-1, -1),    
    )
    MAX_INVENTORY = 50
    MIN_INVENTORY = -50

    def __init__(self, data, n_levels, horizon):
        self.t = 0
        self.episode = 0
        self.n_levels = n_levels
        self.data = data
        self.inventory = 0
        self.theta_a = None
        self.theta_b = None
        self.orders = deque()
        self.alpha = 0.5
        self.horizon = horizon
    
    def sample_order_volume(self):
        size = 5
        return size, size    
    
    # Getters
    def get_ref_price(self):
        return self.data.iloc[self.t].values[4*self.n_levels]
    def get_lob_t(self):
        return self.data.iloc[self.t].values[:4*self.n_levels]
    def get_mid_price_movement(self):
        return self.data['Mid Price Movement'].iloc[self.t]
    def get_half_spread(self):
        return self.data['Market Spread'].iloc[self.t] / 2
    def get_market_state(self):
        return self.data.iloc[self.t].values[4*self.n_levels+1:]
    def get_agent_state(self):
        return np.array([self.inventory, self.theta_a, self.theta_b])
    def get_full_state(self):
        return np.concatenate((self.get_market_state(), self.get_agent_state()))
    

    def get_state(self):
        return self.get_full_state()
    
    def reset(self):
        self.t = 0
        self.inventory = 0
        self.theta_a = None
        self.theta_b = None
        self.orders = deque()
        self.alpha = 1
        return self.get_state()


    def clear_inventory(self):
        v = int(self.inventory * self.alpha)
        best_price, side = None, None
        if v < 0:
            # We need to buy
            best_price = self.get_lob_t()[0]
            side = 'bid'
        elif v > 0:
            # We need to sell
            best_price = self.get_lob_t()[2]
            side = 'ask'
        # Add at the beginning of the list
        if best_price is not None and side is not None:
            self.orders.insert(0, (best_price, abs(v), side))
            
        
        

    def match_orders(self):
        lob_t = self.get_lob_t()
        # Check the orders from the oldest to the newest
        transactions_ask = []
        transactions_bid = []

        for i in range(len(self.orders)):
            # price, volume, side = self.orders[i]
            price, volume, side = self.orders.popleft()
            if side == 'bid':
                # We want to buy
                for level in range(self.n_levels):
                    # Get price and volume at this level
                    p_a_lob = lob_t[4 * level]
                    v_a_lob = lob_t[4 * level + 1]
                    if price >= p_a_lob:
                        # We have a match                        
                        match_price = price
                        match_volume = min(volume, v_a_lob)
                        # Track the transaction
                        transactions_bid.append((match_price, match_volume))
                        # Update the inventory
                        self.inventory += match_volume
                        # Update the order
                        volume -= match_volume
                    if volume == 0:
                        break
            else:
                # We want to sell
                for level in range(self.n_levels):
                    # Get price and volume at this level
                    p_b_lob = lob_t[4 * level + 2]
                    v_b_lob = lob_t[4 * level + 3]
                    print('Level', level)
                    if price == p_b_lob:
                        print('Probably market order')
                        print('N levels', self.n_levels)
                        print('Order Price', price)
                        print('Order Volume', volume)
                        print('LOB Price', p_b_lob)
                        print('LOB Volume', v_b_lob)
                    if price <= p_b_lob:
                        # We have a match                        
                        match_price = price
                        match_volume = min(volume, v_b_lob)
                        # Track the transaction
                        transactions_ask.append((match_price, match_volume))
                        # Update the inventory
                        self.inventory -= match_volume
                        # Update the order
                        volume -= match_volume
                    # if volume == 0:
                    #     break
            if volume > 0:
                # The order is not completed
                self.orders.append((price, volume, side))
        return transactions_ask, transactions_bid
    

    def update(self, action):
        assert action in range(len(EnvironmentState.ACTION_SPACE))
        
        self.theta_a, self.theta_b = EnvironmentState.ACTION_SPACE[action]
        ref_price = self.get_ref_price()
        half_spread = self.get_half_spread()
        
        # Do the action
        if self.theta_a < 0 and self.theta_b < 0:
            self.clear_inventory()
        else:
            # Compute the bid and ask prices along with the volume
            p_a = ref_price + (half_spread * self.theta_a)
            p_b = ref_price + (half_spread * self.theta_b)
            # Check if the inventory is within the limits
            if self.inventory >= EnvironmentState.MIN_INVENTORY and self.inventory <= EnvironmentState.MAX_INVENTORY:
                v_a, v_b = self.sample_order_volume()
                # Place the orders
                self.orders.append((p_a, v_a, 'ask'))
                self.orders.append((p_b, v_b, 'bid'))
            else:
                # Clear the inventory
                self.theta_a, self.theta_b = -1, -1
                self.clear_inventory()

        if self.inventory > EnvironmentState.MAX_INVENTORY or self.inventory < EnvironmentState.MIN_INVENTORY:
            print('Orders')
            for order in self.orders:
                print('\t', order)
            print('.....')
            print('LOB')
            print(self.get_lob_t())
            print('......')
            print('Inventary before matching orders', self.inventory)
        
        # Check if there are matching orders
        transactions_ask, transactions_bid = self.match_orders() # Inventory can change inside this function

        # LOGGING
        if self.inventory > EnvironmentState.MAX_INVENTORY or self.inventory < EnvironmentState.MIN_INVENTORY:
            print("Inventory out of bounds at time ", self.t)
            print("Inventory after transactions", self.inventory)
            print("Ask transactions\n", transactions_ask, end='\n\n')
            print("Bid transactions\n", transactions_bid, end='\n-----\n')


        # Compute phi_a and phi_b
        phi_a = 0
        bankroll = 0
        for price, volume in transactions_ask:
            phi_a += volume * (price - ref_price)
            bankroll += volume * price
        phi_b = 0
        for price, volume in transactions_bid:
            phi_b += volume * (ref_price - price)
            bankroll -= volume * price
            
        # Update the time
        self.t += 1
            
        # Compute inventory portion of reward
        mid_price_change = self.get_mid_price_movement()
        psi = mid_price_change * self.inventory
        # Compute psi dampening factor
        psi = PsiDampening.asymm_damp(psi, 0.5)
        # Compute the reward
        reward = phi_a + phi_b + psi



        return self.get_state(), reward, (self.t == self.horizon), bankroll
