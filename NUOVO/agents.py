import numpy as np
import random
import torch


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
        q_values = self.q_value_approximator.predict(state).cpu().detach().numpy()
        print('Q-values', q_values)
        return np.argmax(q_values)  # Exploit
    
    def behaviour_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        else:
            return self.target_policy(state)  # Exploit
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def train(self, env, n_episodes):
        rewards = np.zeros(n_episodes)
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                print('Time', env.t)
                # Choose action based on the behaviour policy
                if env.t > 0:
                    action = self.behaviour_policy(state)
                else:
                    action = np.random.randint(self.action_size)
                    # Update the state - update theta_a and theta_b
                    theta_a, theta_b = env.ACTION_SPACE[action]
                    state[-2] = theta_a
                    state[-1] = theta_b
                    
                print('Action', action)
                
                # Take action and observe next state and reward
                next_state, reward, done = env.update(action)
                print('Next state', next_state)
                
                # Learn from the transition
                if not done:
                    self.learn(state, action, reward, next_state)
                
                # Update the state
                state = next_state
                total_reward += reward if not done else 0
                
            # Update epsilon
            self.update_epsilon()
            
            # Store the total reward
            rewards[episode] = total_reward
        
    def learn(self, state, action, reward, next_state):
        # Combine the current state and action
        x = np.concatenate([state, [action]]).astype(np.float32)
        print('State', x)
        x = torch.tensor(x, dtype=torch.float32).to(self.q_value_approximator.model.device)
        
        # Compute y_pred
        y_pred = self.q_value_approximator.predict(x)
        
        # Compute y_true
        preds = np.zeros(self.action_size)
        for a in range(self.action_size):
            x_next = np.concatenate([next_state, [a]])
            x_next = torch.tensor(x_next, dtype=torch.float32).to(self.q_value_approximator.model.device)
            preds[a] = self.q_value_approximator.predict(x_next).item()
        y_true = [reward + self.gamma * np.max(preds)]
        print('Y_true', y_true)
        
        # Update the Q-value approximator - Returns the loss
        _ = self.q_value_approximator.update(y_true, y_pred)
        
        
       