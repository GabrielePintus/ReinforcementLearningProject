import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import deque
from tqdm.notebook import tqdm



def plot_rewards(rewards, title, sma=10):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards.rolling(sma).mean(), label='Rewards')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()
    
def play(agent, env, n_episodes):
    policy_function = agent.policy if hasattr(agent, 'policy') else agent.target_policy
    
    episodes_frames = []
    episodes_rewards = []
    
    for _ in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        episode_frames = []
        episode_reward = 0
        
        steps = 0
        while not done and steps < 1_000:
            action = policy_function(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
            episode_frames.append(env.render())
            steps += 1
        
        episodes_frames.append(episode_frames)
        episodes_rewards.append(episode_reward)
        
    return episodes_frames, episodes_rewards


# class ReplayBuffer:
#     def __init__(self, batch_size, buffer_size):
#         self.batch_size = batch_size
#         self.buffer_size = buffer_size
#         self.buffer = deque(maxlen=self.buffer_size)
        
#     def push(self, state, action, reward, next_state, done):
#         experience = (state, action, reward, next_state, done)
#         self.buffer.append(experience)
        
#     def sample(self):
#         batch = random.sample(self.buffer, self.batch_size)
#         state, action, reward, next_state, done = map(np.stack, zip(*batch))
#         return state, action, reward, next_state, done
    

class ReplayBuffer:
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.counts = np.zeros(self.buffer_size)
        self.current_size = 0
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.current_size += 1
        
    def sample(self):
        probs = np.exp(self.counts[:self.current_size]) / np.sum(np.exp(self.counts[:self.current_size]))
        batch = random.choices(self.buffer, weights=probs, k=self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done    
    
    
    
if __name__ == '__main__':
    buffer = ReplayBuffer(10, 100)
    for i in range(100):
        buffer.push(np.random.randn(3), np.random.randint(3), np.random.randn(1), np.random.randn(3), np.random.randint(2))
    state, action, reward, next_state, done = buffer.sample()
    print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)