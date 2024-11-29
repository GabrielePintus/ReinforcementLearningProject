import numpy as np
import random
import torch
from environment import EnvironmentState
from tqdm.auto import tqdm


class LearningAgent:
    def __init__(
        self,
        action_size, # 10 in our case
        q_value_approximator, 
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        gamma=0.99,
        logger=None
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

        assert logger is not None, "Logger must be provided"
        self.logger = logger
        

    #
    #       POLICIES
    #
    def random_policy(self):
        """
        Choose a random action.
        """
        return random.randint(0, self.action_size - 1)
    
    def target_policy(self, state):
        """
        Choose the best action according to the target policy.
        """
        # For each action, predict the Q-value
        q_values = np.zeros(self.action_size)
        for action in range(10):
            # Set thetas
            state_action = self._combine_state_action(state, action)
            # Predict Q-value
            q_values[action] = self.q_value_approximator.predict(state_action)
        # Return the action with the highest Q-value
        return np.argmax(q_values)
    
    def behaviour_policy(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Explore
            return self.random_policy()
        else:
            # Exploit
            return self.target_policy(state)
        

    #
    #       Updates
    #
    def update_epsilon(self):
        """
        Update the exploration rate.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



    #
    #       Learning
    #
    def learn(self, buffer):
        """
        Learn from a batch of transitions.
        """
        X, y = zip(*buffer)
        X, y = np.array(X), np.array(y)

        # Update the Q-value approximator
        loss = self.q_value_approximator.update(X, y)

        return loss

    # 
    #    UTILITY FUNCTIONS
    # 
    def _combine_state_action(self, state, action):
        """
        Combine the state and action into a single array.
        """
        state_action = np.copy(state)
        state_action[-2] = EnvironmentState.ACTION_SPACE[action][0]
        state_action[-1] = EnvironmentState.ACTION_SPACE[action][1]
        return np.concatenate([state_action, [action]]).astype(np.float32)


    #
    #       TRAINING
    #
    def train(self, env, n_repetitions=1, buffer_size=25):
        """
        Train the agent using the given environment.

        Parameters:
        - env (Environment): The environment to train the agent on.
        - n_episodes (int): The number of episodes to train the agent for.
        - buffer_size (int): The size of the buffer to store transitions.
        """

        losses = []
        bankrolls = []
        rewards = []

        for _ in range(n_repetitions):
            # Reset the environment
            state = env.reset()
            done = False

            # Initialize the buffer for storing transitions
            buffer = []
            
            while not done:
                # Choose an action based on the behaviour policy (epsilon-greedy)
                action = self.behaviour_policy(state)
                
                # Take action and observe next state and reward
                next_state, reward, done, bankroll = env.update(action)

                rewards.append(reward)
                bankrolls.append(bankroll)

                # Choose the best action according to the target policy
                best_action = self.target_policy(next_state)
                best_state_action = self._combine_state_action(next_state, best_action)
                # Predict the Q-value
                target = reward + self.gamma * self.q_value_approximator.predict(best_state_action)

                # Learn from the transition
                if len(buffer) == buffer_size:
                    loss = self.learn(buffer)
                    losses.append(loss)
                    buffer = []
                    # buffer.pop(0)
                    # buffer.append((state_action, target))
                else:
                    state_action = self._combine_state_action(state, action)
                    buffer.append((state_action, target))
                
                # Update the state
                state = next_state
                
            # Update epsilon
            self.update_epsilon()
            
        
        return rewards, losses, bankrolls
        
       