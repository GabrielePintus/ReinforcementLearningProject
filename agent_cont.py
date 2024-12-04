from agent import LearningAgent
import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict

from discretizer import GridSpace




class QAgent(LearningAgent):

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            learning_rate_decay=0.99,
            min_learning_rate=0.1,
            seed = 0,
            action_gridspace = None,
            state_gridspace = None
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, learning_rate_decay, min_learning_rate, seed)

        # Q-values
        # self.q_values = np.zeros((state_gridspace.n_states, action_gridspace.n_states))
        self.q_values = np.random.normal(0, 1, (state_gridspace.n_states, action_gridspace.n_states))

        # GridSpace
        self.action_gridspace = action_gridspace
        self.state_gridspace = state_gridspace


    def behaviour_policy(self, state):
        """
        Return a random action.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.action_gridspace.decode(self.target_policy(state))
    
    def target_policy(self, state):
        """
        Return the greedy action.
        """
        return np.argmax(self.q_values[state])
        

    def learn(self, n_episodes=1000, horizon=np.inf):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Episodes')
        for _ in progress_bar:
            # Get initial state and encode it
            state, _ = self.env.reset()
            state = self.state_gridspace.encode(state)

            done = False
            cum_reward = 0
            t = 0
            while not done and t < horizon:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward

                # Encode action, state and next state
                next_state = self.state_gridspace.encode(next_state)
                action = self.action_gridspace.encode(action)

                # Compute the best action for the next state
                next_action = self.target_policy(next_state)

                # Encode next action
                # next_action = self.action_gridspace.encode(next_action)
                
                # Compute the target
                target = reward + self.discount_factor * self.q_values[next_state, next_action]
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.q_values[state, action] += delta * self.learning_rate
                state = next_state
                t += 1

            self.update_epsilon()
            self.update_learning_rate()
            rewards.append(cum_reward)
            progress_bar.set_postfix({'Reward': cum_reward, 'Epsilon': self.epsilon, 'Learning Rate': self.learning_rate})

        return rewards
    
    def play(self, n_episodes=100, horizon=np.inf, render=False):
        """
        Play the game.
        """
        if render:
            self.env = self.env.unwrapped
            self.env.render_mode = 'human'

        rewards = []
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            state = self.state_gridspace.encode(state)

            done = False
            cum_reward = 0
            t = 0
            while not done and t < horizon:
                action = self.target_policy(state)
                next_state, reward, done, _, _ = self.env.step(self.action_gridspace.decode(action))
                cum_reward += reward
                state = self.state_gridspace.encode(next_state)
                t += 1

                if render:
                    self.env.render()

            rewards.append(cum_reward)

        return rewards
    

class QLambdaAgent(LearningAgent):

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed = 0,
            action_gridspace = None,
            state_gridspace = None,
            trace_decay = 0.9,
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        # GridSpace
        self.action_gridspace = action_gridspace
        self.state_gridspace = state_gridspace

        # Q-values
        self.q_values = np.zeros((state_gridspace.n_states, action_gridspace.n_states))

        # Eligibility Trace
        self.trace_decay = trace_decay
        self.eligibility_trace = np.zeros((state_gridspace.n_states, action_gridspace.n_states))

    # def target_policy(self, state):
    #     """
    #     Return the greedy action.
    #     """
    #     return np.argmax(self.q_values[state])

    def behaviour_policy(self, state):
        """
        Return a random action.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.action_gridspace.decode(self.target_policy(state))

    def learn(self, n_episodes=1000, horizon=np.inf):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Episodes')
        for _ in progress_bar:
            # Get initial state and encode it
            state, _ = self.env.reset()
            state = self.state_gridspace.encode(state)

            action = self.behaviour_policy(state)
            action = self.action_gridspace.encode(action)

            done = False
            cum_reward = 0
            t = 0

            while not done and t < horizon:
                # Take action and observe reward and next state
                action = self.action_gridspace.decode(action)
                next_state, reward, done, _, _ = self.env.step(action)

                # Encode action, state and next state
                next_state = self.state_gridspace.encode(next_state)
                action = self.action_gridspace.encode(action)

                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)
                next_action = self.action_gridspace.encode(next_action)

                # Compute best action for next state
                next_best_action = self.target_policy(next_state)
                next_best_action = self.action_gridspace.encode(next_best_action)

                # Compute the target
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace
                self.eligibility_trace[state, action] = self.eligibility_trace[state, action] * 0.9 + 1

                # Update Q-values
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
                t += 1

            self.update_epsilon()
            self.update_learning_rate()
            rewards.append(cum_reward)
            progress_bar.set_postfix({'Reward': cum_reward, 'Epsilon': self.epsilon})

        return rewards


class QSpatialLambdaAgent(LearningAgent):

    def __init__(
            self,
            env : gym.Env,
            discount_factor=0.99,
            initial_epsilon=0.5,
            epsilon_decay=0.97,
            min_epsilon=0.0,
            learning_rate=0.9,
            seed = 0,
            action_gridspace = None,
            state_gridspace = None,
            trace_decay = 0.9,
            kernel = lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2)
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        # GridSpace
        self.action_gridspace = action_gridspace
        self.state_gridspace = state_gridspace

        # Q-values
        self.q_values = np.zeros((state_gridspace.n_states, action_gridspace.n_states))

        # Eligibility Trace
        self.trace_decay = trace_decay
        self.kernel = kernel
        self.eligibility_trace = np.zeros((state_gridspace.n_states, action_gridspace.n_states))

    # def target_policy(self, state):
    #     """
    #     Return the greedy action.
    #     """
    #     return np.argmax(self.q_values[state])

    def behaviour_policy(self, state):
        """
        Return a random action.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.action_gridspace.decode(self.target_policy(state))

    def learn(self, n_episodes=1000, horizon=np.inf):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Episodes')
        for _ in progress_bar:
            # Get initial state and encode it
            state, _ = self.env.reset()
            state = self.state_gridspace.encode(state)

            action = self.behaviour_policy(state)
            action = self.action_gridspace.encode(action)

            done = False
            cum_reward = 0
            t = 0

            while not done and t < horizon:
                # Take action and observe reward and next state
                action = self.action_gridspace.decode(action)
                next_state, reward, done, _, _ = self.env.step(action)

                # Encode action, state and next state
                next_state = self.state_gridspace.encode(next_state)
                action = self.action_gridspace.encode(action)

                cum_reward += reward

                # Take action from Behaviour Policy
                next_action = self.behaviour_policy(next_state)
                next_action = self.action_gridspace.encode(next_action)

                # Compute best action for next state
                next_best_action = self.target_policy(next_state)
                next_best_action = self.action_gridspace.encode(next_best_action)

                # Compute the target
                target = reward + (self.discount_factor * self.q_values[next_state, next_best_action])
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace
                self.eligibility_trace[state, action] += 1

                # Update Q-values
                for s, a in np.ndindex(self.q_values.shape):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[s, a]
                    if next_action == next_best_action:
                        self.eligibility_trace[s, a] *= self.discount_factor * self.trace_decay
                    else:
                        # self.eligibility_trace[s, a] = 0
                        self.eligibility_trace[s, a] *= self.kernel(self.state_gridspace.decode(s), self.state_gridspace.decode(state))
                    # Reset Eligibility Trace if it is too small
                    if self.eligibility_trace[s, a] < 1e-3:
                        self.eligibility_trace[s, a] = 0


                state = next_state
                action = next_action
                t += 1

            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({'Reward': cum_reward, 'Epsilon': self.epsilon})

        return rewards
