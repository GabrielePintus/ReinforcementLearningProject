import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from tqdm import tqdm
import os

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

def init_weights(m):
    if isinstance(m, nn.Linear):  # Apply to linear layers
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:  # Ensure bias exists before initializing
            nn.init.zeros_(m.bias)

class SACAgent(LearningAgent):
    def __init__(
            self, 
            env, 
            discount_factor = 0.99, 
            initial_epsilon = None, 
            epsilon_decay = None, 
            min_epsilon = None, 
            learning_rate = None, 
            lr_actor = 3e-4, 
            lr_critic_value = 3e-4, 
            tau=0.005, 
            max_size=1000000, 
            batch_size=256, 
            reward_scale=2,
            seed=0,
            save_weights = False,
            load_weights = False
        ):
        '''
        env: the environment to interact with.
        discount_factor: the discount factor for future rewards.
        learning_rate: the learning rate for the Q-value approximator.
        lr_actor: the learning rate for the actor network.
        lr_critic_value: the learning rate for the critic network and value network.
        tau: the soft update parameter for the target networks.
        max_size: the maximum size of the replay buffer.
        batch_size: the batch size for training the networks.
        reward_scale: the scaling factor for the rewards.
        seed: the seed for reproducibility.
        save_weights: whether to save the networks' weights.
        load_weights: whether to load the networks' weights.
        '''

        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.save_weights = save_weights
        self.load_weights = load_weights

        # Replay Buffer
        self.memory = ReplayBuffer(max_size = max_size,
                                   input_shape = [env.observation_space.shape[0]],
                                   n_actions = env.action_space.shape[0])

        # Actor-Critic Networks
        self.actor = ActorNetwork(lr = lr_actor,
                                  input_dim = [env.observation_space.shape[0]], 
                                  n_actions = env.action_space.shape[0], 
                                  hidden1_dim = 256,
                                  hidden2_dim = 256, 
                                  max_action=env.action_space.high,
                                  name ='actor',
                                  chkpt_dir = 'tmp/sac')
        
        self.critic_1 = CriticNetwork(lr = lr_critic_value,
                                      input_dim = env.observation_space.shape[0], 
                                      n_actions = env.action_space.shape[0],
                                      hidden1_dim = 256,
                                      hidden2_dim = 256,
                                      name='critic_1',
                                      chkpt_dir = 'tmp/sac')

        self.critic_2 = CriticNetwork(lr = lr_critic_value,
                                      input_dim = env.observation_space.shape[0], 
                                      n_actions = env.action_space.shape[0], 
                                      hidden1_dim = 256,
                                      hidden2_dim = 256,
                                      name='critic_2',
                                      chkpt_dir = 'tmp/sac')

        self.value = ValueNetwork(lr = lr_critic_value,
                                  input_dim = [env.observation_space.shape[0]],
                                  hidden1_dim = 256,
                                  hidden2_dim = 256,
                                  name='value',
                                  chkpt_dir = 'tmp/sac')
        
        self.target_value = ValueNetwork(lr = lr_critic_value,
                                         input_dim = [env.observation_space.shape[0]],
                                         hidden1_dim = 256,
                                         hidden2_dim = 256,
                                         name = 'target_value',
                                         chkpt_dir = 'tmp/sac')

        self.update_network_parameters(tau=1)  # Initialize target network


        self.actor.apply(init_weights)
        self.critic_1.apply(init_weights)
        self.critic_2.apply(init_weights)
        self.value.apply(init_weights)
        self.target_value.apply(init_weights)

    def choose_action(self, state):
        """
        Overrides the epsilon-greedy behaviour policy to use the learned SAC policy.
        """

        state = torch.tensor([state], dtype=torch.float32).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Stores transitions in the replay buffer.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        Updates the target network parameters with soft updates.
        """
        if tau is None:
            tau = self.tau

        target_value_params = dict(self.target_value.named_parameters())
        value_params = dict(self.value.named_parameters())

        for name in value_params:
            value_params[name] = tau * value_params[name].clone() + \
                                 (1 - tau) * target_value_params[name].clone()

        self.target_value.load_state_dict(value_params)

    def learn(self, n_episodes=250):
        """
        Implements the SAC learning algorithm.
        """

        if(self.load_weights):
            self.load_models()

        episode_rewards = []
        avg_scores = []
        episode_steps = []

        for i in range(n_episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0

            while not terminated and not truncated:
                    
                action = self.choose_action(state)

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                self.remember(state, action, reward, new_state, terminated)

                self._learn_step()

                state = new_state
                steps += 1

            episode_rewards.append(total_reward)
            avg_score = np.mean(episode_rewards[-25:])
            episode_steps.append(steps)
            avg_scores.append(avg_score)

            print('episode ', i, 'total_reward %.1f' % total_reward, 'avg_reward %.1f' % avg_score, 'steps', steps)

        if(self.save_weights):
            self.save_models()

        return episode_rewards, avg_scores, episode_steps

    def _learn_step(self):
        """
        Executes a single learning step (updates the actor and critic networks).
        """
        if self.memory.memory_counter < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)

        # Update Value Network
        value = self.value(states).view(-1)
        value_ = self.target_value(next_states).view(-1)
        value_[dones] = 0.0

        actions_pred, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1(states, actions_pred)
        q2_new_policy = self.critic_2(states, actions_pred)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)


        value_loss.backward()
        self.value.optimizer.step()

        # Update Actor Network
        actions_pred, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1(states, actions_pred)
        q2_new_policy = self.critic_2(states, actions_pred)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = actor_loss.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        self.actor.optimizer.step()

        # Update Critic Networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.reward_scale * rewards + self.discount_factor * value_
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update target value network
        self.update_network_parameters()

    def save_models(self):
        """
        Save the actor and critic networks' parameters.
        """

        # check that the directory exists
        if(not os.path.exists('tmp/sac')):
            os.makedirs('tmp/sac')

        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
    
    def load_models(self):
        """
        Load the actor and critic networks' parameters.
        """

        # check that the directory exists
        if(not os.path.exists('tmp/sac')):
            raise Exception("No weights to load. Please train the model first.")

        try:
            self.actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.value.load_checkpoint()
            self.target_value.load_checkpoint()
        except:
            raise Exception("Error loading weights. Please train the model first.")

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
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
        self.env.close()
        return rewards