import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from src.networks import Qnet
from src.utils import ReplayBuffer

    

# Import defaultdict
from collections import defaultdict


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
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        total_steps = 0
        for _ in progress_bar:
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            steps = 0
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
            total_steps += steps
            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

        return rewards, total_steps


class QAgentME(LearningAgent):
    def __init__(
        self,
        env: gym.Env,
        discount_factor=0.99,
        initial_epsilon=0.5,
        epsilon_decay=0.97,
        min_epsilon=0.0,
        learning_rate=0.9,
        seed=0
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        np.random.seed(seed)

        # Q-value array (dense). If your state space is huge, you might want a dictionary here.
        self.q_values = defaultdict(float)

    def target_policy(self, state):
        """
        Choose the best action according to the Q-values.
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
        if np.random.rand() < self.epsilon:
            return self.random_policy()
        else:
            return self.target_policy(state)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards = []
        total_steps = 0
        progress_bar = tqdm(range(n_episodes), desc='Simulating QAgent')

        for _ in progress_bar:
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            steps = 0

            while not done and steps < horizon:
                # Epsilon-greedy action
                action = self.behaviour_policy(state)

                next_state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward

                # Greedy action in the next state
                next_action = self.target_policy(next_state)

                # Q-update (TD(0) style)
                td_target = reward + self.discount_factor * self.q_values[next_state, next_action]
                td_error = td_target - self.q_values[state, action]
                self.q_values[state, action] += self.learning_rate * td_error

                state = next_state
                steps += 1

            total_steps += steps
            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'avg_reward_10': np.mean(rewards[-10:])
            })

        return rewards, total_steps





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
        total_steps = 0
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

            total_steps += steps
            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

        return rewards, total_steps
    
    
    
class QLambdaAgentME(LearningAgent):
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
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        
        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        np.random.seed(seed)

        # Q-value approximator (dense). 
        # For extremely large environments, consider a dictionary for Q as well.
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))

        # Eligibility trace: use a sparse dict {(state, action): float}
        self.eligibility_trace = defaultdict(float)

    def target_policy(self, state):
        """Choose the best action from Q-values."""
        return np.argmax(self.q_values[state])

    def random_policy(self):
        """Choose a random action."""
        return self.env.action_space.sample()

    def behaviour_policy(self, state):
        """
        Epsilon-greedy exploration.
        """
        if np.random.rand() < self.epsilon:
            return self.random_policy()
        else:
            return self.target_policy(state)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy using Q(lambda).
        """
        rewards = []
        total_steps = 0
        progress_bar = tqdm(range(n_episodes), desc='Simulating QLambda')

        for _ in progress_bar:
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            steps = 0

            # Clear the eligibility trace dictionary at the start of each episode
            self.eligibility_trace.clear()

            while not done and steps < horizon:
                next_state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward

                next_action = self.behaviour_policy(next_state)
                next_best_action = self.target_policy(next_state)

                # TD error
                td_target = reward + self.discount_factor * self.q_values[next_state, next_best_action]
                delta = td_target - self.q_values[state, action]

                # 1) Increment eligibility for (state, action)
                self.eligibility_trace[(state, action)] += 1.0

                # 2) [Optional] Normalization step (commented out, can be expensive)
                #    If you really need it, do something like:
                #
                q_vals = np.array(list(self.eligibility_trace.values()))
                q_min, q_max = np.min(q_vals), np.max(q_vals)
                if q_max - q_min > 1e-3:
                    for key in self.eligibility_trace:
                        self.eligibility_trace[key] = (
                            (self.eligibility_trace[key] - q_min) / (q_max - q_min)
                        )

                # 3) Update Q-values for all active traces
                for (s, a) in list(self.eligibility_trace.keys()):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[(s, a)]

                # 4) Decay the eligibility traces
                #    - If next_action == next_best_action, we multiply by discount_factor * trace_decay
                #    - Otherwise, we multiply by 0 (in Watkins' Q(lambda)) or do something else
                #      as needed for your definition. The code below follows your "mask" approach:
                factor = (1.0 if next_action == next_best_action else 0.0)
                factor *= self.discount_factor * self.trace_decay

                # Multiply all trace entries by factor
                for (s, a) in list(self.eligibility_trace.keys()):
                    self.eligibility_trace[(s, a)] *= factor
                    # Optionally remove small traces
                    if abs(self.eligibility_trace[(s, a)]) < 1e-5:
                        del self.eligibility_trace[(s, a)]

                state = next_state
                action = next_action
                steps += 1

            total_steps += steps
            self.update_epsilon()
            rewards.append(cum_reward)

            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'avg_reward_10': np.mean(rewards[-10:])
            })

        return rewards, total_steps    

    


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



class QSpatialLambdaAgentME(LearningAgent):
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
        self.env = env
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.kernel = kernel

        # Learning rate
        self.learning_rate = learning_rate
        
        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        np.random.seed(seed)

        # Q-value approximator (dense or dict—here we keep it dense for simplicity)
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))

        # SPARSE eligibility trace stored as a dictionary: { (state, action): float }
        self.eligibility_trace = defaultdict(float)

    def target_policy(self, state):
        """Select action with the highest Q-value for the given state."""
        return np.argmax(self.q_values[state])

    def random_policy(self):
        """Sample a random action."""
        return self.env.action_space.sample()

    def behaviour_policy(self, state):
        """
        Epsilon-greedy policy:
         - with probability epsilon, pick a random action
         - otherwise pick the best action from Q-values
        """
        if np.random.rand() < self.epsilon:
            return self.random_policy()
        else:
            return self.target_policy(state)

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        total_steps = 0
        for _ in progress_bar:
            state, _ = self.env.reset()
            action = self.behaviour_policy(state)
            done = False
            cum_reward = 0
            steps = 0

            # Clear dictionary at the start of each episode
            self.eligibility_trace.clear()

            while not done and steps < horizon:
                next_state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward

                next_action = self.behaviour_policy(next_state)
                next_best_action = self.target_policy(next_state)

                # TD target and TD error
                target = reward + self.discount_factor * self.q_values[next_state, next_best_action]
                delta = target - self.q_values[state, action]

                # 1) Increment eligibility for (state, action)
                self.eligibility_trace[(state, action)] += 1.0

                # 2) [Optional] Normalize the eligibility_trace dictionary if needed
                #    This is expensive, so do it only if essential.
                min_val = min(self.eligibility_trace.values())
                max_val = max(self.eligibility_trace.values())
                if max_val - min_val > 1e-3:
                    for key in self.eligibility_trace:
                        self.eligibility_trace[key] = (
                            (self.eligibility_trace[key] - min_val) / (max_val - min_val)
                        )

                # 3) Update Q-values for all active (s, a) in the eligibility trace
                for (s, a) in list(self.eligibility_trace.keys()):
                    self.q_values[s, a] += self.learning_rate * delta * self.eligibility_trace[(s, a)]

                # 4) Decay the eligibility traces
                #    - If next_action == next_best_action, multiply by discount_factor*trace_decay
                #    - Otherwise, also apply the 'spatial kernel' factor
                if next_action == next_best_action:
                    # Simple case: multiply all by discount_factor * trace_decay
                    for (s, a) in list(self.eligibility_trace.keys()):
                        self.eligibility_trace[(s, a)] *= self.discount_factor * self.trace_decay
                        # Optionally remove small traces
                        if abs(self.eligibility_trace[(s, a)]) < 1e-5:
                            del self.eligibility_trace[(s, a)]
                else:
                    # Harder case: multiply by discount_factor * trace_decay * kernel(...)
                    # We'll compute the kernel factor for each (s,a) in the dict, not for the entire NxM space.
                    for (s, a) in list(self.eligibility_trace.keys()):
                        k = self.kernel(np.array([s, a]), np.array([next_state, next_action]))
                        self.eligibility_trace[(s, a)] *= (self.discount_factor * self.trace_decay * k)
                        if abs(self.eligibility_trace[(s, a)]) < 1e-5:
                            del self.eligibility_trace[(s, a)]

                # Move to next step
                state = next_state
                action = next_action
                steps += 1

            total_steps += steps
            # Decay epsilon each episode
            self.update_epsilon()
            rewards.append(cum_reward)

            # Update the progress bar with some stats
            progress_bar.set_postfix({
                'epsilon': self.epsilon,
                'reward': np.mean(rewards[-10:])
            })

        return rewards, total_steps




class DQNAgent(LearningAgent):
    
    def __init__(
        self,
        env: gym.Env,
        discount_factor=0.99,
        initial_epsilon=0.5,
        epsilon_decay=0.97,
        min_epsilon=0.0,
        learning_rate=0.9,
        seed=0,
        q_main : Qnet = None,
        q_target : Qnet = None,
        batch_size=64,
        buffer_size=10_000,
        intertia=0.99,
    ):
        super().__init__(
            env,
            discount_factor,
            initial_epsilon,
            epsilon_decay,
            min_epsilon,
            learning_rate,
            seed
        )
        # Netowrks update params
        self.intertia = intertia
        
        # Replay buffer
        self.buffer = ReplayBuffer(batch_size, buffer_size)
        
        # Q-value approximator
        self.q_main = q_main
        self.q_target = q_target
        
    
    def update_target(self):
        """
        Update the target network.
        """
        # Get the main and target weights
        main_weights = self.q_main.state_dict()
        target_weights = self.q_target.state_dict()
        
        # Compute and set the new target weights
        target_weights = {
            k: self.intertia * target_weights[k] + (1 - self.intertia) * main_weights[k]
            for k in main_weights
        }
        self.q_target.load_state_dict(target_weights)
        
    def target_policy(self, state):
        """
        Choose the best action according to the target policy.
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.q_target.device)
        self.q_target.eval()
        
        q_values = np.zeros(self.env.action_space.n)
        with torch.no_grad():
            for action in range(self.env.action_space.n):
                q_values[action] = self.q_target(state).detach().cpu().numpy()
        
        return np.argmax(q_values)
    
    
    def train_main(self, gradient_steps=1):
        losses = []
        
        for step in range(gradient_steps):
            # Sample a batch of experiences
            states, actions, rewards, next_states, dones = self.buffer.sample()
            states       = torch.tensor(states, dtype=torch.float32).to(self.q_main.device)
            actions      = torch.tensor(actions, dtype=torch.long).to(self.q_main.device)
            rewards      = torch.tensor(rewards, dtype=torch.float32).to(self.q_main.device)
            next_states  = torch.tensor(next_states, dtype=torch.float32).to(self.q_main.device)
            dones        = torch.tensor(dones, dtype=torch.float32).to(self.q_main.device)
            
            with torch.no_grad():
                # Compute next Q-values
                next_q_values = self.q_target(next_states)
                next_q_values, _ = torch.max(next_q_values, dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step td target
                target_q_values = rewards + (1-dones) * self.discount_factor * next_q_values
                
            # Get current Q-values estimates
            current_q_values = self.q_main(states)
            
            # Compute the loss
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            
            # Optimize the model
            self.q_main.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), 10)
            self.q_main.optimizer.step()
            
        return losses
            
            
        
    
    
    def learn(self, n_episodes=1000, horizon=10_000):
        """
        Learn the optimal policy.
        """
        rewards, losses = [], []
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for episode in progress_bar:
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            steps = 0
            episode_losses = []

            while not done and steps < horizon:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward
                
                # Compute the Q-values
                # best_action = self.target_policy(next_state)

                # Store experience in the replay buffer
                self.buffer.push(state, action, reward, next_state, done)
                
                # Train the Main Q-network
                _losses = self.train_main()
                episode_losses.append(np.mean(_losses))
                
            # Update the target network
            self.update_target()
            
            self.update_epsilon()
            rewards.append(cum_reward)
            losses.append(episode_losses)
        
        return rewards, losses