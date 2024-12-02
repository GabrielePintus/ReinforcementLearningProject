import gymnasium as gym
import numpy as np
from tqdm import tqdm


class LearningAgent:

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed = 0
        ):
        self.env = env
        self.discount_factor = discount_factor
        self.seed = seed
        self.learning_rate = learning_rate

        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def target_policy(self, state):
        """
        Choose the best action according to the target policy.
        """
        return np.argmax(self.q_values[state, :])
    
    def random_policy(self):
        """
        Choose a random action.
        """
        return self.env.action_space.sample()
    
    def behaviour_policy(self, state):
        """
        Epsilon-greedy policy for exploration.
        """
        action = self.random_policy() if np.random.rand() < self.epsilon else self.target_policy(state)
        return action
    
    def update_epsilon(self):
        """
        Update epsilon value.
        """
        # Linear decay over episodes
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.min_epsilon, epsilon)


    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        raise NotImplementedError

    def play(self, n_episodes=1, render = False):
        """
        Play the game using the learned policy.
        """
        
        if(render):
            self.env = self.env.unwrapped
            self.env.render_mode = 'human'

        rewards = []
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.target_policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
        return rewards





class QAgent(LearningAgent):

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed = 0
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))

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

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)

                # Lazyness penalty
                # if not done:
                #     reward += -0.1

                # Check if the agent has fallen into a hole
                if done and reward == 0:
                    reward = -1

                cum_reward += reward

                # Compute the best action for the next state
                next_action = self.target_policy(next_state)
                
                target = reward + self.discount_factor * self.q_values[next_state, next_action]
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.q_values[state, action] += delta * self.learning_rate
                state = next_state
                steps += 1
            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards
    




class QLambdaAgent(LearningAgent):

    def __init__(
            self,
            env: gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed=0,
            trace_decay=0.9
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        self.trace_decay = trace_decay
        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.eligibility_trace = np.zeros((env.observation_space.n, env.action_space.n))


    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            max_steps = 100
            steps = 0

            while not done and steps < max_steps:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)

                # Lazyness penalty
                # if not done:
                #     reward += -0.1

                # Check if the agent has fallen into a hole
                if done and reward == 0:
                    reward = -1

                # Compute the best action for the next state
                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)

                # Comput best action for next state
                next_best_action = self.target_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.eligibility_trace[state, action] += 1
                # self.eligibility_trace[state, action] = max(self.eligibility_trace[state, action], 1)

                for s, a in np.ndindex(self.q_values.shape):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[s, a]
                    if next_action == next_best_action:
                        self.eligibility_trace[s, a] *= self.discount_factor * self.trace_decay
                    else:
                        self.eligibility_trace[s, a] = 0
                    # Reset Eligibility Trace if it is too small
                    if self.eligibility_trace[s, a] < 1e-3:
                        self.eligibility_trace[s, a] = 0


                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards




class SarsaAgent(LearningAgent):

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

        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            max_steps = np.inf
            steps = 0

            while not done and steps < max_steps:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)

                # Lazyness penalty
                # if not done:
                #     reward += -0.1

                # Check if the agent has fallen into a hole
                if done and reward == 0:
                    reward = -1

                # Compute the best action for the next state
                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_action])
                delta = target - self.q_values[state, action]

                # Update Q-values
                self.q_values[state, action] += self.learning_rate * delta

                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards


class SarsaLambdaAgent(LearningAgent):

    def __init__(
            self,
            env: gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed=0,
            trace_decay=0.9
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)
        
        self.trace_decay = trace_decay

        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.eligibility_trace = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            max_steps = 100
            steps = 0

            while not done and steps < max_steps:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)

                # Lazyness penalty
                # if not done:
                #     reward += -0.05

                # Check if the agent has fallen into a hole
                if done and reward == 0:
                    reward = -1

                # Sum the reward
                cum_reward += reward

                # Compute the next action according to the behaviour policy
                next_action = self.behaviour_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.eligibility_trace[state, action] += 1
                # self.eligibility_trace[state, action] = max(self.eligibility_trace[state, action], 1)

                for s, a in np.ndindex(self.q_values.shape):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[s, a]
                    self.eligibility_trace[s, a] *= self.discount_factor * self.trace_decay
                    # Reset Eligibility Trace if it is too small
                    if self.eligibility_trace[s, a] < 1e-3:
                        self.eligibility_trace[s, a] = 0

                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards
            



class QSpatialLambdaAgent(LearningAgent):

    def __init__(
            self,
            env: gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed=0,
            trace_decay=0.9,
            kernel = lambda x, y: np.exp(-np.linalg.norm(x - y))
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        self.trace_decay = trace_decay
        self.kernel = kernel

        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.eligibility_trace = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, n_episodes=1000):
        """
        Learn the optimal policy.
        """
        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            max_steps = 100
            steps = 0

            while not done and steps < max_steps:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)

                # Lazyness penalty
                # if not done:
                #     reward += -0.1

                # Check if the agent has fallen into a hole
                if done and reward == 0:
                    reward = -1

                # Compute the best action for the next state
                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)

                # Comput best action for next state
                next_best_action = self.target_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.eligibility_trace[state, action] += 1
                # self.eligibility_trace[state, action] = max(self.eligibility_trace[state, action], 1)

                for s, a in np.ndindex(self.q_values.shape):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[s, a]
                    if next_action == next_best_action:
                        self.eligibility_trace[s, a] *= self.discount_factor * self.trace_decay
                    else:
                        # Decay the trace using gaussian kernel
                        value = self.kernel(np.array([s, a]), np.array([next_state, next_action]))
                        self.eligibility_trace[s, a] *= value
                    # Reset Eligibility Trace if it is too small
                    if self.eligibility_trace[s, a] < 1e-3:
                        self.eligibility_trace[s, a] = 0


                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)

        return rewards


