import pandas as pd

class DataGenerator:
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
        levels (int): Number of levels in the order book
    """
    @staticmethod
    def generator(filename, levels = 1): # CHANGE LEVELS TO 5 QUANDO USEREMO IL FILE "VERO"
        df = pd.read_parquet(filename)

        data = {}
        for level in range(1, levels + 1):
            data[f'Ask Price {level}'] = df.iloc[:, (level - 1) * 4] / 1e4       # QULACUNO NEL MERCATO VUOLE COMPRARE A TOT
            data[f'Ask Volume {level}'] = df.iloc[:, (level - 1) * 4 + 1]   # QUALCUNO NEL MERCATO VUOLE COMPRARE TOT QUANTITA'
            data[f'Bid Price {level}'] = df.iloc[:, (level - 1) * 4 + 2] / 1e4   # QUALCUNO NEL MERCATO VUOLE VENDERE A TOT
            data[f'Bid Volume {level}'] = df.iloc[:, (level - 1) * 4 + 3]  # QUALCUNO NEL MERCATO VUOLE VENDERE TOT QUANTITA'
        
        # Crea un nuovo DataFrame con le colonne riorganizzate
        LOB = pd.DataFrame(data)

        LOB['Mid Price'] = (LOB['Ask Price 1'] + LOB['Bid Price 1']) / 2
        LOB['Market Spread'] = LOB['Ask Price 1'] - LOB['Bid Price 1']
        LOB['Mid Price Movement'] = LOB['Mid Price'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]).fillna(0)
        LOB['Book Imbalance'] = (LOB['Bid Volume 1'] - LOB['Ask Volume 1']) / (LOB['Bid Volume 1'] + LOB['Ask Volume 1'])
        LOB['Signed Volume'] = LOB['Bid Volume 1'] - LOB['Ask Volume 1']

        # Issues:

        # Returns = (Price at time t - Price at time t-1) / Price at time t-1
        LOB['Returns'] = LOB['Mid Price'].pct_change()

        # Volatility è la std dei returns, che non abbiamo, su 10 periodi. Quindi usiamo come proxy il moving mid price?
        LOB['Volatility'] = LOB['Returns'].rolling(window=10).std()

        delta = LOB['Mid Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        LOB['RSI'] = 100 - (100 / (1 + rs))

        # RSI ha bisogno di gain e loss, che non abbiamo. Qui ho usato come gain e loss le variazioni positive e negative del mid price a mo' di proxy. Va bene?
        # RSI=100 − 100/(1+ RS)
        # RS = Average Gain / Average Loss per un periodo di tempo (di solito 14 istanze)

        return LOB.dropna().reset_index(drop=True).drop(columns=['Returns'])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    df = DataGenerator.generator('data/lob.csv', levels=1)
    
    # Plot mid price
    print(df.head())

    


