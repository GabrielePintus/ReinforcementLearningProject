import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm


class LearningAgent:

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            learning_rate_decay=0.99995,
            min_learning_rate=0.1,
            seed = 0
        ):
        self.env = env
        self.discount_factor = discount_factor
        self.seed = seed
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate

        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def target_policy(self, state):
        """
        Choose the best action according to the target policy.
        """
        return np.argmax(self.q_values[state])
    
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

    def update_learning_rate(self):
        """
        Update learning rate.
        """
        # Linear decay over episodes
        learning_rate = self.learning_rate * self.learning_rate_decay
        self.learning_rate = max(self.min_learning_rate, learning_rate)


    def learn(self):
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

    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards = []
        steps = 0
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for _ in progress_bar:
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            while not done and steps < horizon:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                
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
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

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


    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for episode in progress_bar:
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            steps = 0

            while not done and steps < horizon:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Compute the best action for the next state
                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)

                # Comput best action for next state
                next_best_action = self.target_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and normalize it
                self.eligibility_trace[state, action] += 1
                et_min = np.min(self.eligibility_trace)
                et_max = np.max(self.eligibility_trace)
                if et_max - et_min > 1e-3:
                    self.eligibility_trace -= et_min
                    self.eligibility_trace /= (et_max - et_min)

                # Update Q-values
                self.q_values += self.learning_rate * delta * self.eligibility_trace
                
                # Apply the trace decay where the next action is the best action                
                self.eligibility_trace *= (next_action == next_best_action) * self.discount_factor * self.trace_decay
                
                # Set to zero the trace where < 1e-3
                # self.eligibility_trace[self.eligibility_trace < 1e-3] = 0

                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

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
            kernel = lambda x, y : np.exp(-np.linalg.norm(x - y) ** 2)
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        self.trace_decay = trace_decay
        self.kernel = kernel

        # Q-value approximator
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.eligibility_trace = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for episode in progress_bar:
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            steps = 0

            while not done and steps < horizon:
                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                    
                # Compute the best action for the next state
                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)

                # Comput best action for next state
                next_best_action = self.target_policy(next_state)

                # Compute the TD error
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Normalize it
                self.eligibility_trace[state, action] += 1
                et_min = np.min(self.eligibility_trace)
                et_max = np.max(self.eligibility_trace)
                if et_max - et_min > 1e-3:
                    self.eligibility_trace -= et_min
                    self.eligibility_trace /= (et_max - et_min)
                    
                # Update Q-values
                self.q_values += self.learning_rate * delta * self.eligibility_trace
  
                # Apply the trace decay where the next action is the best action
                mask = (next_action == next_best_action)
                self.eligibility_trace *= mask * self.discount_factor * self.trace_decay
                
                # Apply kernel spatial decay where the next action is not the best action
                if next_action != next_best_action and np.sum(self.eligibility_trace) > 1e-3:
                    # Create a grid of state-action pairs
                    state_action_pairs = np.array(np.meshgrid(np.arange(self.q_values.shape[0]), np.arange(self.q_values.shape[1]))).T.reshape(-1, 2)
                    next_state_action = np.array([next_state, next_action])
                    
                    # Compute the kernel values for all state-action pairs
                    kernel_values = np.apply_along_axis(lambda pair: self.kernel(pair, next_state_action), 1, state_action_pairs)
                    
                    # Reshape kernel values to match the shape of eligibility_trace
                    kernel_values = kernel_values.reshape(self.q_values.shape)
                    
                    # Apply the kernel values to decay the eligibility trace
                    self.eligibility_trace *= kernel_values

                # Set to zero the trace where < 1e-3
                # self.eligibility_trace[self.eligibility_trace < 1e-3] = 0

                state = next_state
                action = next_action
                steps += 1

            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

        return rewards


