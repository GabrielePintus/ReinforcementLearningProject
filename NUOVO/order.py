

class Order:

    ID = 0

    def __init__(self, price, volume, side, market=False, t=-1):
        self.id = Order.ID
        Order.ID += 1
        self.price = price
        self.volume = volume
        self.side = side
        self.market = market

        # Useful for debugging
        self.t = t

    def __str__(self):
        text =  f'--| ID: {self.id}, Time: {self.t} |--\n'
        text += f'    Price: {self.price}, Volume: {self.volume}, Side: {self.side}, Market: {self.market}'
        return text

    def match(self, other):
        if self.side == 'ask' and other.side == 'bid':
            return self.price <= other.price
        elif self.side == 'bid' and other.side == 'ask':
            return self.price >= other.price
        else:
            return False
        
    



class Transaction:

    def __init__(self, order, transaction_volume, level, time):
        self.order = order
        self.transaction_volume = transaction_volume
        self.level = level
        self.time = time

