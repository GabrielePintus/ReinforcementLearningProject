from agent import LearningAgent
import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm
# from collections import defaultdict

from discretizer import TilingEncoder
from scipy.sparse import csr_matrix, dok_matrix



class custom_array(dok_matrix):
    
    def __init__(self, encoder, *args, **kwargs):
        self.encoder = encoder
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, key):
        # If key is a slice of none this means that we are selecting all the elements
        # of the array. In this case we return the array itself
        key = self.encoder(*key)
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        key = self.encoder(*key)
        super().__setitem__(key, value)
        
    def __contains__(self, key):
        key = self.encoder(*key)
        return super().__contains__(key)
    
    def __iter__(self):
        for key in super().__iter__():
            yield key


class QAgent(LearningAgent):

    def __init__(
            self,
            env: gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed=0,
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        # Tile encoders
        self.state_encoder = TilingEncoder(
            bounds=np.array([[-1.2, 0.6], [-0.07, 0.07]]),
            tiles=20,
            single_idx=True
        )
        self.state_space_n = 20 ** 2
        self.action_encoder = TilingEncoder(
            bounds=np.array([[-1.0, 1.0]]),
            tiles=20,
            single_idx=True
        )
        self.action_space_n = 20
        
        self.encoder = lambda x, y : (self.state_encoder(x), self.action_encoder(y))

        # Sparse Q-value approximator
        # self.q_values = dok_matrix((env.observation_space.n, env.action_space.n), dtype=np.float32)
        self.q_values = custom_array(self.encoder, (self.state_space_n, self.action_space_n), dtype=np.float32)



    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        rewards = []
        max_steps = 100
        steps = 0
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            while not done and steps < max_steps:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)
                # state = self.state_encoder(state)

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                # action = self.action_encoder(action)
                # next_state = self.state_encoder(next_state)

                # Lazyness penalty
                # if not done:
                #     reward += -0.1

                # Check if the agent has fallen into a hole
                # if done and reward == 0:
                #     reward = -1

                cum_reward += reward

                # Compute the best action for the next state
                next_action = self.target_policy(next_state)
                # next_action = self.action_encoder(next_action)
                
                target = reward + self.discount_factor * self.q_values[next_state, next_action]
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                # self.q_values[state, action] += delta * self.learning_rate
                self.q_values[state, action] = self.q_values[state, action] + delta * self.learning_rate
                state = next_state
                steps += 1
            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards
    
