from tabulate import tabulate


class LOB:

    def __init__(self, data, n_levels):
        self.data = data
        self.n_levels = n_levels
        self.t = 0

    def best_bid_price(self):
        return self.data.iloc[self.t].values[2]
    def best_ask_price(self):
        return self.data.iloc[self.t].values[0]
    
    def step(self):
        self.t += 1
        return self.data.iloc[self.t].values
    
    def get_state(self):
        return self.data.iloc[self.t].values[:4*self.n_levels]
    
    def get_mid_price(self):
        return self.data.iloc[self.t].values[4*self.n_levels]
    def get_ref_price(self):
        return self.get_mid_price()
    def get_mid_price_movement(self):
        return self.data['Mid Price Movement'].iloc[self.t]
    def get_half_spread(self):
        return self.data['Market Spread'].iloc[self.t] / 2
    
    def get_ask(self, level):
        price = self.data.iloc[self.t].values[4*level]
        volume = self.data.iloc[self.t].values[4*level+1]
        return price, volume
    
    def get_bid(self, level):
        price = self.data.iloc[self.t].values[4*level+2]
        volume = self.data.iloc[self.t].values[4*level+3]
        return price, volume
        
    def get_features(self):
        return self.data.iloc[self.t].values[4*self.n_levels+1:]
    



if __name__ == '__main__':
    from data import DataGenerator

    data = DataGenerator.generator("data/AAPL.parquet", levels=10).head(30000)
    lob = LOB(data, 10)
    lob.step()


    
        