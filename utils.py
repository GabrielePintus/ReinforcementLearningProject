import matplotlib.pyplot as plt

def plot_rewards(rewards, title, sma=10):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards.rolling(sma).mean(), label='Rewards')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()
    
    
from collections import deque
import random 
import numpy as np




class ReplayBuffer:
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
if __name__ == '__main__':
    buffer = ReplayBuffer(10, 100)
    for i in range(100):
        buffer.push(np.random.randn(3), np.random.randint(3), np.random.randn(1), np.random.randn(3), np.random.randint(2))
    state, action, reward, next_state, done = buffer.sample()
    print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)