import gymnasium as gym
import numpy as np


class LearningAgent:
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        initial_epsilon=0.5,
        epsilon_decay=0.97,
        min_epsilon=0.0,
        q_value_approximator=None,
        seed = 0
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.seed = seed
        
        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay  
        self.min_epsilon = min_epsilon
        
        # Q-value approximator
        self.q_value_approximator = q_value_approximator
        
    
    #
    #    POLICIES
    #   
    def random_policy(self):
        """
        Choose a random action.
        """
        return self.env.action_space.sample()
    
    def target_policy(self, state):
        """
        Choose the best action according to the target policy.
        """
        raise NotImplementedError
    
    def behaviour_policy(self, state):
        """
        Epsilon-greedy policy for exploration.
        """
        action = self.random_policy() if np.random.rand() < self.epsilon else self.target_policy(state)
        return np.array(action)
        
    def update_epsilon(self):
        """
        Update epsilon value.
        """
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.min_epsilon, epsilon)
        
    #
    #    LEARNING
    #
    
    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        raise NotImplementedError
    
    def evaluate(self, n_episodes=100):
        """
        Evaluate the agent's performance.
        """
        raise NotImplementedError
    
    def play(self, n_episodes=1):
        """
        Play the game using the learned policy.
        """
        raise NotImplementedError
    
    
        
    

#
#      Q-LEARNING AGENT
#
from sklearn.mixture import GaussianMixture

class QLearningAgent(LearningAgent):
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        initial_epsilon=0.5,
        epsilon_decay=0.97,
        min_epsilon=0.0,
        q_value_approximator=None,
        seed = 0  
    ):
        super().__init__(
            env,
            discount_factor,
            initial_epsilon,
            epsilon_decay,
            min_epsilon,
            q_value_approximator,
            seed
        )
        
        # Place a Gaussian Mixture Model on the action space
        self.n_components = 2
        self.n_dims = env.action_space.shape[0]
        self.action_space_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.seed,
            init_params='random'
        )
        # Initialize the model
        self.action_space_model.fit(np.zeros((self.n_components, self.n_dims)))
        
    
    def target_policy(self, state, n_samples=10):
        """
        Choose the best action according to the target policy.
        """
        # Sample actions from the action space model
        actions, _ = self.action_space_model.sample(n_samples)
        
        # Compute Q-values for each action
        q_values = np.zeros(n_samples)
        for i in range(n_samples):
            action = actions[i]
            sa_couple = np.concatenate([state, action], axis=0)
            q_values[i] = self.q_value_approximator.predict(sa_couple)
            
        # Return the action with the highest Q-value
        return actions[np.argmax(q_values)]
    
    
    def train(self, state, buffer=[], buffer_size=16):
        """
        Train the agent.
        """
        # Sample action from the behaviour policy
        action = self.behaviour_policy(state)
        sa_couple = np.concatenate([state, action], axis=0)
        
        # Take action and observe reward and next state
        # next_state, reward, done, truncated, info = self.env.step(action)
        next_state, reward, done, _, _ = self.env.step(action)
        
        # Compute the best action for the next state
        next_action = self.target_policy(next_state)
        next_sa_couple = np.concatenate([next_state, next_action], axis=0)
        
        # Compute the target Q-value
        target = reward + self.discount_factor * self.q_value_approximator.predict(next_sa_couple)
        
        # Update the Q-value approximator
        if len(buffer) < buffer_size:
            buffer.append((sa_couple, target))
        else:
            X = np.array([x[0] for x in buffer])
            y = np.array([x[1] for x in buffer])
            self.q_value_approximator.update(X, y)
        
        return next_state, reward, done
        
        
        
    
    