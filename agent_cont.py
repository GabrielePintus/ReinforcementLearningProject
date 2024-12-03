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
            seed = 0,
            action_gridspace = None,
            state_gridspace = None
        ):
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, min_epsilon, learning_rate, seed)

        # Q-values
        self.q_values = defaultdict(lambda: 0)

        # GridSpace
        self.action_gridspace = action_gridspace
        self.state_gridspace = state_gridspace

    def target_policy(self, state):
        """
        Return the greedy action.
        """
        # The set of possible action is the cartesian product of each row in the action grid space codomain
        actions = np.array(np.meshgrid(*[np.arange(i, j) for i, j in self.action_gridspace.codomain])).T.reshape(-1, len(self.action_gridspace.codomain))
        q_values = np.array([self.q_values[state, tuple(action)] for action in actions])
        return actions[np.argmax(q_values)]
        
        

    def learn(self, n_episodes=1000, horizon=np.inf):
        """
        Learn the optimal policy.
        """
        rewards = []
        progress_bar = tqdm(range(n_episodes), desc='Episodes')
        for _ in progress_bar:
            # Get initial state and encode it
            state, _ = self.env.reset()
            state = tuple(self.state_gridspace.encode(np.array(state)))

            done = False
            cum_reward = 0
            t = 0
            while not done and t < horizon:
                # Sample action from the behaviour policy
                action = self.behaviour_policy(state)

                # Take action and observe reward and next state
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Check if the next state is the flag
                # if done and state[0] >= 0.5:
                #     pass
                # else:
                #     reward -= 100

                # Encode action, state and next state
                next_state = tuple(self.state_gridspace.encode(np.array(next_state)))
                action = tuple(self.action_gridspace.encode(action))

                cum_reward += reward

                # Compute the best action for the next state
                next_action = self.target_policy(next_state)

                # Encode next action
                next_action = tuple(self.action_gridspace.encode(next_action))
                
                # Compute the target
                target = reward + self.discount_factor * self.q_values[next_state, next_action]
                delta = target - self.q_values[state, action]

                # Update Eligibility Trace and Q-values
                self.q_values[state, action] += delta * self.learning_rate
                state = next_state
                t += 1

            self.update_epsilon()
            rewards.append(cum_reward)
            progress_bar.set_postfix({'Reward': cum_reward, 'Epsilon': self.epsilon})

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
            state = tuple(self.state_gridspace.encode(np.array(state)))
            done = False
            cum_reward = 0
            t = 0
            while not done and t < horizon:
                action = self.target_policy(state)
                state, reward, done, _, _ = self.env.step(action)
                state = tuple(self.state_gridspace.encode(np.array(state)))
                cum_reward += reward
                if render:
                    self.env.render()
                if cum_reward > horizon:
                    break
                t += 1
            rewards.append(cum_reward)
        return rewards
    

