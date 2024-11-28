import numpy as np
import random
import torch
from states import EnvironmentState
from tqdm.auto import tqdm


class LearningAgent:
    def __init__(
        self,
        action_size, # 10 in our case
        q_value_approximator, 
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        gamma=0.99
    ):
        """
        Initialize the Learning Agent.

        Parameters:
        - action_size (int): The size of the action space.
        - q_value_approximator (QValueApproximator): The Q-value approximator model.
        - epsilon (float): Initial exploration rate for epsilon-greedy strategy.
        - epsilon_decay (float): Decay factor for exploration rate.
        - min_epsilon (float): Minimum value for exploration rate.
        - gamma (float): Discount factor for future rewards.
        """
        self.action_size = action_size
        self.q_value_approximator = q_value_approximator
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        
    def target_policy(self, state):
        # For each action, predict the Q-value and return the action with the highest Q-value
        q_values = np.zeros(self.action_size)
        for action in range(10):
            state[-2] = EnvironmentState.ACTION_SPACE[action][0]
            state[-1] = EnvironmentState.ACTION_SPACE[action][1]
            x = np.concatenate([state, [action]]).astype(np.float32)
            q_values[action] = self.q_value_approximator.predict(x)
        return np.argmax(q_values)  # Exploit
    
    def behaviour_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        else:
            return self.target_policy(state)  # Exploit
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    # def train(self, env, n_episodes, buffer_size=20):
    #     rewards = np.zeros(n_episodes)
    #     bankrolls = np.zeros(n_episodes)
    #     losses = []
        
    #     # for episode in tqdm(range(n_episodes)):
    #     for episode in range(n_episodes):
    #         state = env.reset()
    #         done = False
    #         total_reward = 0
    #         total_bankroll = 0
            
    #         while not done:
    #             # print('Time', env.t)
    #             # Choose action based on the behaviour policy
    #             if env.t > 0:
    #                 action = self.behaviour_policy(state)
    #             else:
    #                 action = np.random.randint(self.action_size)
    #                 # Update the state - update theta_a and theta_b
    #                 theta_a, theta_b = env.ACTION_SPACE[action]
    #                 state[-2] = theta_a
    #                 state[-1] = theta_b
                
    #             # Take action and observe next state and reward
    #             next_state, reward, done, bankroll = env.update(action)
                
    #             # Learn from the transition
    #             if not done:
    #                 loss = self.learn(state, action, reward, next_state)
    #                 losses.append(loss)
                
    #             # Update the state
    #             state = next_state
    #             total_reward += reward if not done else 0
    #             total_bankroll += bankroll if not done else 0
                
    #         # Update epsilon
    #         self.update_epsilon()
            
    #         # Store the total reward
    #         rewards[episode] = total_reward
    #         bankrolls[episode] = total_bankroll
        
    #     return rewards, losses, bankrolls

    def train(self, env, n_episodes=1, buffer_size=16):
        rewards = np.zeros(n_episodes)
        bankrolls = np.zeros(n_episodes)
        losses = []
        
        # for episode in tqdm(range(n_episodes)):
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            total_bankroll = 0
            buffer = []
            
            while not done:
                # print('Time', env.t)
                # Choose action based on the behaviour policy
                if env.t > 0:
                    action = self.behaviour_policy(state)
                else:
                    action = np.random.randint(self.action_size)
                    # Update the state - update theta_a and theta_b
                    theta_a, theta_b = env.ACTION_SPACE[action]
                    state[-2] = theta_a
                    state[-1] = theta_b
                
                # Take action and observe next state and reward
                next_state, reward, done, bankroll = env.update(action)

                # Compute target value
                q_values = np.zeros(self.action_size)
                for a in range(self.action_size):
                    q_values[a] = self.q_value_approximator.predict(np.concatenate([next_state, [a]]).astype(np.float32))
                target = reward + self.gamma * np.max(q_values)
                
                # Learn from the transition
                if len(buffer) == buffer_size:
                    X, y = zip(*buffer)
                    X = np.array(X)
                    y = np.array(y)
                    loss = self.q_value_approximator.update(X, y)
                    losses.append(loss)
                    buffer = []
                    # # Remove first element
                    # buffer.pop(0)
                    # # Add the new element
                    # buffer.append((np.concatenate([state, [action]]).astype(np.float32), target))
                else:
                    buffer.append((np.concatenate([state, [action]]).astype(np.float32), target))
                
                # Update the state
                state = next_state
                total_reward += reward if not done else 0
                total_bankroll += bankroll if not done else 0
                
            # Update epsilon
            self.update_epsilon()
            
            # Store the total reward
            rewards[episode] = total_reward
            bankrolls[episode] = total_bankroll
        
        return rewards, losses, bankrolls
        
    # def learn(self, state, action, reward, next_state):
    #     # Combine the current state and action
    #     x = np.concatenate([state, [action]]).astype(np.float32)
        
    #     # Compute y_pred
    #     y_pred = self.q_value_approximator.predict(x)

    #     # Compute y_true
    #     preds = np.zeros(self.action_size)
    #     # X contains all the states and actions
    #     X = np.zeros((self.action_size, len(x)))
    #     for a in range(self.action_size):
    #         x_next = np.concatenate([next_state, [a]])
    #         preds[a] = self.q_value_approximator.predict(x_next)
    #     y_true = [reward + self.gamma * np.max(preds)]
        
    #     # Update the Q-value approximator - Returns the loss
    #     loss = self.q_value_approximator.update(X, y_true)

    #     return loss

        
        
       