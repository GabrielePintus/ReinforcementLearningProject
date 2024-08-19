import pandas as pd

df = pd.read_csv('data/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv', header=None)

levels = 1

data = {}
for level in range(1, levels + 1):
    data[f'Ask Price {level}'] = df.iloc[:, (level - 1) * 4]
    data[f'Ask Volume {level}'] = df.iloc[:, (level - 1) * 4 + 1]
    data[f'Bid Price {level}'] = df.iloc[:, (level - 1) * 4 + 2]
    data[f'Bid Volume {level}'] = df.iloc[:, (level - 1) * 4 + 3]

LOB = pd.DataFrame(data)

LOB['Mid Price'] = (LOB['Ask Price 1'] + LOB['Bid Price 1']) / 2
LOB['Market Spread'] = LOB['Ask Price 1'] - LOB['Bid Price 1']

LOB['Mid Price Movement'] = LOB['Mid Price'].diff().shift(-1) # Che è anche il return
LOB['Book Imbalance'] = (LOB['Bid Volume 1'] - LOB['Ask Volume 1']) / (LOB['Bid Volume 1'] + LOB['Ask Volume 1'])
LOB['Signed Volume'] = LOB['Bid Volume 1'] - LOB['Ask Volume 1']
LOB['Volatility'] = LOB['Mid Price Movement'].rolling(window=10).std()

# Calcola il Relative Strength Index (RSI) (usando una finestra mobile di 14 periodi)
delta = LOB['Mid Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
LOB['RSI'] = 100 - (100 / (1 + rs))

# Come fare RSI?
# RSI=100 − 100/(1+ RS)
# RS = Average Gain / Average Loss per un periodo di tempo (di solito 14 istanze)


# Visualizza il DataFrame riorganizzato
print(LOB)