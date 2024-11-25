import numpy as np
import pandas as pd

# Custom modules
from data import DataGenerator


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
    MAX_INVENTORY = 10000
    MIN_INVENTORY = -10000

    def __init__(self, data, n_levels):
        self.t = 0
        self.n_levels = n_levels
        self.data = data
        self.inventory = 0
        self.theta_a = None
        self.theta_b = None
        self.orders = []
    
    def sample_order_volume(self):
        size = 1
        return size, size    
    # Getters
    def get_ref_price(self):
        return self.data.iloc[self.t].values[4*self.n_levels]
    def get_lob_t(self):
        return self.data.iloc[self.t].values[:4*self.n_levels]
    def get_mid_price_movement(self):
        return self.data['Mid Price Movement'].iloc[self.t]
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
        self.orders = []
        return self.get_state()


    def match_orders(self, lob_t):
        n_levels = lob_t.shape[1] // 4
        # for i in range(len(self.orders)):
        # Check the orders from the oldest to the newest
        transactions_ask = []
        transactions_bid = []
        for i in range(len(self.orders) - 1, -1, -1):
            price, volume, side = self.orders[i]
            if side == 'bid':
                # We want to buy
                for level in n_levels:
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
                        # Update the order in the list
                        self.orders[i] = (price, volume, side)
                        # Check if the order is completed
                        if volume == 0:
                            # Remove the order
                            self.orders.pop(i)
            else:
                # We want to sell
                for level in n_levels:
                    # Get price and volume at this level
                    p_b_lob = lob_t[4 * level + 2]
                    v_b_lob = lob_t[4 * level + 3]
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
                        # Update the order in the list
                        self.orders[i] = (price, volume, side)
                        # Check if the order is completed
                        if volume == 0:
                            # Remove the order
                            self.orders.pop(i)
        return transactions_ask, transactions_bid
    

    def update(self, action, half_spread):
        assert action in range(len(EnvironmentState.ACTION_SPACE))
        self.theta_a, self.theta_b = EnvironmentState.ACTION_SPACE[action]
        ref_price = self.get_ref_price()
        past_inventory = self.inventory
        
        # Do the action
        if self.theta_a < 0 and self.theta_b < 0:
            # Clear inventory
            pass
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
                pass
        
        # Check if there are matching orders
        transactions_ask, transactions_bid = self.match_orders(self.get_lob_t())
        # Compute phi_a and phi_b
        phi_a = 0
        for price, volume in transactions_ask:
            phi_a += volume * (price - ref_price)
        phi_b = 0
        for price, volume in transactions_bid:
            phi_b += volume * (ref_price - price)
        # Compute inventory portion of reward
        mid_price_change = self.get_mid_price_movement()
        psi = mid_price_change * past_inventory
        # Compute psi dampening factor
        psi = PsiDampening.asymm_damp(psi, 0.5)
        # Compute the reward
        reward = phi_a + phi_b + psi

        # Update the time
        self.t += 1

        return reward
