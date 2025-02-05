import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from tqdm.notebook import tqdm
from copy import deepcopy

from src.Q_agents import LearningAgent
from src.networks import Qnet
from src.utils import ReplayBuffer


class DQNAgent(LearningAgent):
    
    def __init__(
        self,
        env: gym.Env,
        discount_factor=0.99,
        initial_epsilon=0.5,
        epsilon_decay=0.97,
        min_epsilon=0.0,
        learning_rate=0.9,
        q_main : Qnet = None,
        q_target : Qnet = None,
        batch_size=64,
        buffer_size=10_000,
        inertia=0.99,
    ):
        super().__init__(
            env,
            discount_factor,
            initial_epsilon,
            epsilon_decay,
            min_epsilon,
            learning_rate,
        )
        # Netowrks update params
        self.inertia = inertia
        
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
            k: self.inertia * target_weights[k] + (1 - self.inertia) * main_weights[k]
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
            q_values = self.q_target(state).detach().cpu().numpy()
        
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
                rewards = rewards.reshape(-1, 1)
                dones = dones.reshape(-1, 1)
                target_q_values = rewards + (1-dones) * self.discount_factor * next_q_values
                
            # Get current Q-values estimates
            current_q_values = self.q_main(states)
            
            # Compute the loss
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item() / len(states))
            
            # Optimize the model
            self.q_main.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), 0.5)
            self.q_main.optimizer.step()
            
        return losses

    def update_epsilon(self):
        """
        Update the epsilon value.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        
    
    def learn(self, n_episodes=1000, horizon=10_000, warmup=100):
        """
        Learn the optimal policy.
        """
        rewards, losses, steps = [], [], []
        progress_bar = tqdm(range(warmup+n_episodes), desc='Simulating')
        overfitting = False
        overfitting_episode = 0
        for episode in progress_bar:
            state, _ = self.env.reset()
            done = False
            truncated = False
            cum_reward = 0
            episode_steps = 0
            episode_losses = []

            while not done and episode_steps < horizon and not truncated:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)

                # Take action and observe reward and next state
                next_state, reward, done, truncated, _ = self.env.step(action)
                cum_reward += reward

                # Store experience in the replay buffer
                self.buffer.push(state, action, reward, next_state, done)
                
                # Train the Main Q-network
                if len(self.buffer.buffer) > self.buffer.batch_size and not overfitting:
                    _losses = self.train_main()
                    episode_losses.append(np.mean(_losses))
                
                state = next_state
                episode_steps += 1
                
            # Update the target network
            self.update_target()
            if episode > warmup:
                self.update_epsilon()
                
            # Step the learning rate scheduler
            self.q_main.lr_scheduler.step()
            
            rewards.append(cum_reward)
            losses.append(episode_losses)
            steps.append(episode_steps)
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:] if rewards else -np.inf),
                'epsilon': self.epsilon,
                'loss': np.mean([np.mean(l) for l in losses[-10:] if l]),
                'lr': self.q_main.optimizer.param_groups[0]['lr']
            })
        
        return rewards, losses, steps
    
    
    def test(self, n_episodes=100, horizon=10_000, render=False):
        """
        Test the learned policy.
        """
        if render:
            self.env = self.env.unwrapped
            self.env.render_mode = 'rgb_array'
            
        rewards, frames = [], []
        progress_bar = tqdm(range(n_episodes), desc='Testing')
        for episode in progress_bar:
            state, _ = self.env.reset()
            done = False
            cum_reward = 0
            episode_frames = []
            
            while not done and len(episode_frames) < horizon:
                if render:
                    episode_frames.append(np.array(self.env.render()))
                    
                action = self.target_policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                cum_reward += reward
                state = next_state
                
            rewards.append(cum_reward)
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:])
            })
            
            if render:
                frames.append(episode_frames)
                
        return rewards, frames

    def save(self, path):
        """
        Save the network weights.
        """
        torch.save(self.q_target.state_dict(), f'{path}_q-function.pth')
        



class MonteCarloPolicyGradient(LearningAgent):
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        policy_net : torch.nn.Module = None,
        inertia=0.5,
    ):
        self.env = env
        self.discount_factor = discount_factor
        
        # Policy network
        self.policy_net_main = policy_net
        self.policy_net_target = deepcopy(policy_net)
        # Initialize weights
        self.policy_net_main.init_weights(True)
        self.policy_net_target.init_weights(False)
        
        # Update params
        self.inertia = inertia
        
    def policy(self, state):
        self.policy_net_target.eval()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.policy_net_target.device)
        
        self.policy_net_target.eval()
        with torch.no_grad():
            action_probs = self.policy_net_target(state)
        
        action = Categorical(action_probs).sample().item()
            
        return action
    
    def update_target(self):
        main_weights = self.policy_net_main.state_dict()
        target_weights = self.policy_net_target.state_dict()
        
        # Use inertia to update target network weights
        target_weights = {
            k: self.inertia * target_weights[k] + (1 - self.inertia) * main_weights[k]
            for k in main_weights
        }
        self.policy_net_target.load_state_dict(target_weights)
    
    def sample_trajectory(self, horizon=10_000):
        state, _ = self.env.reset()
        done = False
        episode = []
        steps = 0
        
        while not done and steps < horizon:
            action = self.policy(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
            
        return episode
    
    def learn(self, n_episodes=1000, horizon=10_000):
        rewards = []
        losses = []
        total_rewards = []
        
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for _ in progress_bar:
            trajectory = self.sample_trajectory(horizon)
            
            # Compute 
            G = 0
            returns = []
            for _, _, reward in trajectory[::-1]:
                G = reward + self.discount_factor * G
                returns.append(G)
            
            # Move the returns to the device
            returns = torch.tensor(returns[::-1], dtype=torch.float32).to(self.policy_net_main.device)
            states, actions, _ = zip(*trajectory)
            states = torch.tensor(states, dtype=torch.float32).to(self.policy_net_main.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.policy_net_main.device)
            
            self.policy_net_main.train()
            
            action_probs = self.policy_net_main(states)
            action_log_probs = Categorical(action_probs).log_prob(actions)
            
            loss = -torch.sum(action_log_probs * returns)
            losses.append(loss.item())
            
            self.policy_net_main.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net_main.parameters(), 0.5)
            self.policy_net_main.optimizer.step()
            
            rewards.append(sum(r for _, _, r in trajectory))
            
            self.update_target()
            
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:]),
                'loss': np.mean(losses[-10:]),
            })
            
            if np.mean(rewards[-20:]) > 495:
                break
            
        return rewards, losses
        
    
class MonteCarloActorCritic(LearningAgent):
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        policy_net : torch.nn.Module = None,
        value_net : torch.nn.Module = None,
        inertia=0.0,
    ):
        self.env = env
        self.discount_factor = discount_factor
        
        # Policy network
        self.policy_net_main = policy_net
        self.policy_net_target = deepcopy(policy_net)
        # Initialize weights
        self.policy_net_main.init_weights(True)
        self.policy_net_target.init_weights(False)
        
        # Value network
        self.value_net_main = value_net
        self.value_net_target = deepcopy(value_net)
        # Initialize weights
        self.value_net_main.init_weights(True)
        self.value_net_target.init_weights(False)
        
        # Update params
        self.inertia = inertia
        
    def policy(self, state):
        self.policy_net_target.eval()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.policy_net_target.device)
        
        self.policy_net_target.eval()
        with torch.no_grad():
            action_probs = self.policy_net_target(state)
        
        action = Categorical(action_probs).sample().item()
            
        return action
    
    def update_target(self):
        main_weights = self.policy_net_main.state_dict()
        target_weights = self.policy_net_target.state_dict()
        
        # Use inertia to update target network weights
        target_weights = {
            k: self.inertia * target_weights[k] + (1 - self.inertia) * main_weights[k]
            for k in main_weights
        }
        self.policy_net_target.load_state_dict(target_weights)
        
        main_weights = self.value_net_main.state_dict()
        target_weights = self.value_net_target.state_dict()
        
        # Use inertia to update target network weights
        target_weights = {
            k: self.inertia * target_weights[k] + (1 - self.inertia) * main_weights[k]
            for k in main_weights
        }
        self.value_net_target.load_state_dict(target_weights)
    
    def sample_trajectory(self, horizon=10_000):
        state, _ = self.env.reset()
        done = False
        truncated = False
        episode = []
        
        steps = 0
        while not done and not truncated and steps < horizon:
            action = self.policy(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
            
        return episode
    
    def learn(self, n_episodes=1000, horizon=10_000):
        rewards, losses_actor, losses_critic = [], [], []
        
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for _ in progress_bar:
            trajectory = self.sample_trajectory(horizon)
            
            # Compute 
            G = 0
            returns = []
            values = []
            for state, _, reward in trajectory[::-1]:
                G = reward + self.discount_factor * G
                returns.append(G)
                
                # Compute the value of the state
                state = torch.tensor(state, dtype=torch.float32).to(self.value_net_main.device)
                state = state.view(1, -1)
                self.value_net_main.eval()
                value = self.value_net_main(state)
                values.append(value)
                
            # Move the returns to the device
            returns = torch.tensor(returns[::-1], dtype=torch.float32).to(self.value_net_main.device)
            values = torch.tensor(values[::-1], dtype=torch.float32).to(self.value_net_main.device)
            advantages = (returns - values).detach()
            states, actions, _ = zip(*trajectory)
            states = torch.tensor(states, dtype=torch.float32).to(self.policy_net_main.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.policy_net_main.device)
            
            # Train the value network
            self.value_net_main.train()
            self.value_net_main.optimizer.zero_grad()
            value = self.value_net_main(states)
            loss_critic = F.huber_loss(value, returns.view(-1, 1))
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net_main.parameters(), 0.5)
            self.value_net_main.optimizer.step()
            losses_critic.append(loss_critic.item())
            
            # Train the policy network
            self.policy_net_main.train()
            action_probs = self.policy_net_main(states)
            action_log_probs = Categorical(action_probs).log_prob(actions)
            loss_actor = -torch.sum(action_log_probs * advantages)
            losses_actor.append(loss_actor.item())
            self.policy_net_main.optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net_main.parameters(), 0.5)
            self.policy_net_main.optimizer.step()
            
            # Update target networks
            self.update_target()
            
            # Track the states, actions and rewards
            rewards.append(sum(r for _, _, r in trajectory))
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:]),
                'loss actor': np.mean(losses_actor[-10:]),
                'loss critic': np.mean(losses_critic[-10:]),
            })
            
        return rewards, losses_actor, losses_critic
    



class ProximalPolicyOptimization(LearningAgent):
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        policy_net : torch.nn.Module = None,
        value_net : torch.nn.Module = None,
        ratio_clip=0.2,
        entropy_weight=0.0,
    ):
        self.env = env
        self.discount_factor = discount_factor
        
        # Policy network
        self.policy_net = policy_net
        self.value_net = value_net
        # Initialize weights
        self.policy_net.init_weights(True)
        self.value_net.init_weights(True)
        # Old policy network
        self.policy_net_old = deepcopy(policy_net)
                
        # Loss params
        self.ratio_clip = ratio_clip
        self.entropy_weight = entropy_weight
        
        
    def policy(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.policy_net_old.device)
        
        self.policy_net_old.eval()
        with torch.no_grad():
            action_probs = self.policy_net_old(state)
        action = Categorical(action_probs).sample().item()
            
        return action
    
    def critic(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.value_net.device)
        
        self.value_net.eval()
        with torch.no_grad():
            value = self.value_net(state)
            
        return value
    
    
    def sample_trajectory(self, horizon=10_000):
        state, _ = self.env.reset()
        done = False
        truncated = False
        episode = []
        
        steps = 0
        while not done and not truncated and steps < horizon:
            action = self.policy(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
            
        return episode
    
    
    def learn(self, n_episodes=1000, horizon=10_000):
        rewards, losses_actor, losses_critic = [], [], []
        
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for _ in progress_bar:
            trajectory = self.sample_trajectory(horizon)
            
            # Compute returns and advantages
            G = 0
            returns = []
            values = []
            for state, _, reward in trajectory[::-1]:
                # Compute the return
                G = reward + self.discount_factor * G
                returns.append(G)
                
                # Compute the value estimate of the state
                value = self.critic(state)
                values.append(value)
            
            # Compute the advantages
            values = torch.tensor(values[::-1], dtype=torch.float32).to(self.value_net.device)
            returns = torch.tensor(returns[::-1], dtype=torch.float32).to(self.value_net.device)
            advantages = (returns - values).detach()
            
            # Move the states, actions and advantages to the device
            states, actions, _ = zip(*trajectory)
            states = torch.tensor(states, dtype=torch.float32).to(self.policy_net.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.policy_net.device)
            
            # Train the value network
            self.value_net.train()
            self.value_net.optimizer.zero_grad()
            value = self.value_net(states)
            loss_critic = F.huber_loss(value, returns.view(-1, 1))
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_net.optimizer.step()
            losses_critic.append(loss_critic.item())
            
            # Compute the probability ratio with the old policy
            action_probs_old = self.policy_net_old(states)
            action_log_probs_old = Categorical(action_probs_old).log_prob(actions)
            
            # Compute the probability ratio with the current policy
            action_probs = self.policy_net(states)
            action_log_probs = Categorical(action_probs).log_prob(actions)

            # Compute the probability ratio
            ratio = torch.exp(action_log_probs - action_log_probs_old)
            
            # Compute the surrogate clipped loss
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages
            
            # Compute the entropy loss
            entropy = Categorical(action_probs).entropy().mean()
            
            # Compute full loss
            # loss_actor = -torch.min(surrogate1, surrogate2).mean()
            loss_actor = -torch.min(surrogate1, surrogate2).mean() - self.entropy_weight * entropy
            losses_actor.append(loss_actor.item())
            
            # Save the current weights for later
            policy_nn_state = self.policy_net.state_dict().copy()
            
            # Neural network optimization
            self.policy_net.train()
            self.policy_net.optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_net.optimizer.step()
            
            # Update the old policy network
            self.policy_net_old.load_state_dict(policy_nn_state)
            
            # Track the rewards
            rewards.append(sum(r for _, _, r in trajectory))
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:]),
                'loss actor': np.mean(losses_actor[-10:]),
                'loss critic': np.mean(losses_critic[-10:]),
            })
        
        return rewards, losses_actor, losses_critic
    
    

class MiniBatchProximalPolicyOptimization(LearningAgent):
    
    def __init__(
        self,
        env : gym.Env,
        discount_factor=0.99,
        policy_net : torch.nn.Module = None,
        value_net : torch.nn.Module = None,
        ratio_clip=0.2,
        entropy_weight=0.0,
        batch_size=32,
        n_epochs=2
    ):
        self.env = env
        self.discount_factor = discount_factor
        
        # Policy network
        self.policy_net = policy_net
        self.value_net = value_net
        # Initialize weights
        self.policy_net.init_weights(True)
        self.value_net.init_weights(True)
        # Old policy network
        self.policy_net_old = deepcopy(policy_net)
                
        # Loss params
        self.ratio_clip = ratio_clip
        self.entropy_weight = entropy_weight
        
        # Mini-batch params
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        
    def policy(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.policy_net_old.device)
        
        self.policy_net_old.eval()
        with torch.no_grad():
            action_probs = self.policy_net_old(state)
        action = Categorical(action_probs).sample().cpu().numpy()
            
        return action
    
    def critic(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.value_net.device)
        
        self.value_net.eval()
        with torch.no_grad():
            value = self.value_net(state)
            
        return value
    
    
    def sample_trajectory(self, horizon=10_000):
        state, _ = self.env.reset()
        
        done = [ False ] * state.shape[0]
        truncated = [ False ] * state.shape[0]
        episode = []
        
        steps = 0
        # while not done and not truncated and steps < horizon:
        while not any(done) and not any(truncated) and steps < horizon:
            action = self.policy(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
            
        return episode
    
    
    def learn(self, n_episodes=1000, horizon=10_000):
        rewards, losses_actor, losses_critic = [], [], []
        
        progress_bar = tqdm(range(n_episodes), desc='Simulating')
        for _ in progress_bar:
            trajectory = self.sample_trajectory(horizon)
            
            # Compute returns and advantages
            G = 0
            returns = []
            values = []
            for state, _, reward in trajectory[::-1]:
                # Compute the return
                G = reward + self.discount_factor * G
                returns.append(G)
                
                # Compute the value estimate of the state
                value = self.critic(state)
                values.append(value)   
            
            returns = np.array([ r for r in returns[::-1]])
            values = np.array([ v.cpu().numpy() for v in values[::-1]])
            
            # Compute the advantages
            returns = torch.tensor(returns, dtype=torch.float32).to(self.value_net.device)
            values = torch.tensor(values, dtype=torch.float32).to(self.value_net.device).view(returns.shape)
            
            # Compute the advantages
            advantages = (returns - values).detach()
            
            # Move the states, actions and advantages to the device
            states, actions, _ = zip(*trajectory)
            states = torch.tensor(states, dtype=torch.float32).to(self.policy_net.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.policy_net.device)
            
            # Flatten the multiple trajectories
            T, N, *obs_shape = states.shape
            states = states.view(T * N, *obs_shape)     # Now shape: (T*N, obs_dim)
            actions = actions.view(T * N)               # Now shape: (T*N,)
            advantages = advantages.view(T * N)         # Now shape: (T*N,)
            returns = returns.view(T * N)               # Now shape: (T*N,)
            
            # Create the mini batch data loaders
            actor_dataset = torch.utils.data.TensorDataset(states, actions, advantages)
            critic_dataset = torch.utils.data.TensorDataset(states, returns)
            actor_loader = torch.utils.data.DataLoader(actor_dataset, batch_size=self.batch_size, shuffle=True)
            critic_loader = torch.utils.data.DataLoader(critic_dataset, batch_size=self.batch_size, shuffle=True)            
            
            for _ in range(self.n_epochs):
                for states, actions, advantages in actor_loader:
                    # Compute the probability ratio with the old policy
                    action_probs_old = self.policy_net_old(states)
                    action_log_probs_old = Categorical(action_probs_old).log_prob(actions)

                    # Compute the probability ratio with the current policy
                    action_probs = self.policy_net(states)
                    action_log_probs = Categorical(action_probs).log_prob(actions)

                    # Compute the probability ratio
                    ratio = torch.exp(action_log_probs - action_log_probs_old)

                    # Compute the surrogate clipped loss
                    surrogate1 = ratio * advantages
                    surrogate2 = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages

                    # Compute the entropy loss
                    entropy = Categorical(action_probs).entropy().mean()

                    # Compute full loss
                    loss_actor = -torch.min(surrogate1, surrogate2).mean() - self.entropy_weight * entropy
                    losses_actor.append(loss_actor.item())

                    # Save the current weights for later
                    policy_nn_state = self.policy_net.state_dict().copy()

                    # Neural network optimization
                    self.policy_net.train()
                    self.policy_net.optimizer.zero_grad()
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                    self.policy_net.optimizer.step()

                    # Update the old policy network
                    self.policy_net_old.load_state_dict(policy_nn_state)
                    
                for states, returns in critic_loader:
                    # Train the value network
                    self.value_net.train()
                    self.value_net.optimizer.zero_grad()
                    value = self.value_net(states)
                    loss_critic = F.huber_loss(value, returns.view(-1, 1))
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                    self.value_net.optimizer.step()
                    losses_critic.append(loss_critic.item())
                    
            # Track the rewards
            rewards.append(sum(r for _, _, r in trajectory))
            progress_bar.set_postfix({
                'reward': np.mean(rewards[-10:]),
                'loss actor': np.mean(losses_actor[-10:]),
                'loss critic': np.mean(losses_critic[-10:]),
            })
            
        return rewards, losses_actor, losses_critic
    


