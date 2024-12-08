import matplotlib.pyplot as plt

def plot_rewards(rewards, title, sma=10):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards.rolling(sma).mean(), label='Rewards')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()