import numpy as np
import pandas as pd
from collections import deque
from order import Order
from LOB import LOB



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

    def __init__(self, lob, n_levels, horizon):
        self.t = 0
        self.episode = 0
        self.n_levels = n_levels
        self.lob = lob
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
    def get_market_state(self):
        return self.lob.get_features()
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
        self.orders = []
        self.alpha = 1
        return self.get_state()


    def clear_inventory(self):
        """
        Clear the inventory by placing a market order.
        """
        if self.inventory == 0:
            return
        
        # Compute the volume, side, and best price
        v = round(self.inventory * self.alpha)
        side = 'bid' if v < 0 else 'ask'
        best_price = self.lob.best_bid_price() if side == 'bid' else self.lob.best_ask_price()


        # Place the market order
        market_order = Order(best_price, abs(v), side, market=True, t=self.t)
        self.orders.insert(0, market_order)

    def perform_transaction(self, order, transaction_volume):
        """
        Perform a transaction and update the inventory.
        """
        if order.side == 'ask':
            self.inventory -= transaction_volume
            order.volume -= transaction_volume
        elif order.side == 'bid':
            self.inventory += transaction_volume
            order.volume -= transaction_volume
        else:
            raise ValueError('Invalid order side')
        




    def match_orders(self):
        # Check the orders from the oldest to the newest
        transactions_ask = []
        transactions_bid = []

        for i in range(len(self.orders)):
            # Get the order
            order = self.orders.pop(0)

            # Match the order
            if order.side == 'bid':
                # We want to buy
                for level in range(self.n_levels):
                    # Get price and volume at this level
                    p_a_lob, v_a_lob = self.lob.get_ask(level)
                    if order.price >= p_a_lob or order.market:
                        # We have a match                        
                        match_price = order.price
                        match_volume = min(order.volume, v_a_lob)
                        # Track the transaction
                        transactions_bid.append((match_price, match_volume))
                        # Perform the transaction
                        self.perform_transaction(order, match_volume)
                    if order.volume == 0:
                        break
            else:
                # We want to sell
                for level in range(self.n_levels):
                    # Get price and volume at this level
                    p_b_lob, v_b_lob = self.lob.get_bid(level)
                    if order.price <= p_b_lob or order.market:
                        # We have a match                        
                        match_price = order.price
                        match_volume = min(order.volume, v_b_lob)
                        # Track the transaction
                        transactions_ask.append((match_price, match_volume))
                        # Perform the transaction
                        self.perform_transaction(order, match_volume)
                    if order.volume == 0:
                        break
            if order.volume > 0:
                # The order is not completed
                self.orders.append(order)

        # Remove empty orders
        self.orders = [order for order in self.orders if order.volume > 0]

        return transactions_ask, transactions_bid
    

    def _compute_prices(self):
        p_a = self.lob.get_ref_price() + (self.lob.get_half_spread() * self.theta_a)
        p_b = self.lob.get_ref_price() + (self.lob.get_half_spread() * self.theta_b)
        return p_a, p_b

    def update(self, action):
        assert action in range(len(EnvironmentState.ACTION_SPACE))
        
        # Get some useful values
        self.theta_a, self.theta_b = EnvironmentState.ACTION_SPACE[action]
        ref_price   = self.lob.get_ref_price()
        half_spread = self.lob.get_half_spread()
        
        # Do the action
        if self.theta_a < 0 and self.theta_b < 0:
            self.clear_inventory()
        else:
            # Compute the bid and ask prices along with the volume
            p_a, p_b = self._compute_prices()
            # Check if the inventory is within the limits
            if self.inventory >= EnvironmentState.MIN_INVENTORY and self.inventory <= EnvironmentState.MAX_INVENTORY:
                v_a, v_b = self.sample_order_volume()
                # Place the orders
                self.orders.append(Order(p_a, v_a, 'ask', market=False, t=self.t))
                self.orders.append(Order(p_b, v_b, 'bid', market=False, t=self.t))
            else:
                # Clear the inventory
                self.theta_a, self.theta_b = -1, -1
                self.clear_inventory()

        
        # Check if there are matching orders
        transactions_ask, transactions_bid = self.match_orders() # Inventory can change inside this function


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
        mid_price_change = self.lob.get_mid_price_movement()
        psi = mid_price_change * self.inventory
        # Compute psi dampening factor
        psi = PsiDampening.asymm_damp(psi, 0.5)
        # Compute the reward
        reward = phi_a + phi_b + psi



        return self.get_state(), reward, (self.t == self.horizon), bankroll

