

class Order:
    def __init__(self, price, volume, side, market=False, t=-1):
        self.price = price
        self.volume = volume
        self.side = side
        self.market = market

        # Useful for debugging
        self.t = t

    def __str__(self):
        return f'Price: {self.price}, Volume: {self.volume}, Side: {self.side}, Market: {self.market}'

