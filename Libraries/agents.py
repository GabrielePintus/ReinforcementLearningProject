import numpy as np
from tqdm.auto import tqdm
from Libraries import policies
from collections import deque



class LearningAgent:

    def __init__(self, env, value_function, update_rule, policy, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, el_decay=None):   
        self.env = env
        self.value_function = value_function  # Generalized value function approximator
        self.update_rule = update_rule  # Generalized update rule (e.g., Q-learning, SARSA)
        self.policy = policy  # Generalized policy (e.g., epsilon-greedy)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.el_decay = el_decay    # eligibility traces lambda decay term
        self.eligibility_trace = None

        # Initialize eligibility trace for SARSA(λ) if lambda is provided, 
        if el_decay is not None:
            self.eligibility_trace = {}
    
    # Choose action based on the policy
    def choose_action(self, state):
        return self.policy(state, self.value_function, self.env, self.epsilon)
    

    # Update the value function using the specified update rule
    def learn(self, state, action, reward, next_state, next_action, done):
        if self.el_decay is not None:
            self.update_rule(state, action, reward, next_state, next_action, done, self.value_function, self.alpha, self.gamma, self.el_decay, self.eligibility_trace)
        else:
            self.update_rule(state, action, reward, next_state, next_action, done, self.value_function, self.alpha, self.gamma)

    def reset_eligibility_trace(self):
        if self.eligibility_trace is not None:
            self.eligibility_trace = {}
    
    # Decay epsilon - i.e., exploration rate
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Train the agent
    def train(self, n_episodes):
        progress_bar = tqdm(range(n_episodes), desc='Training', unit='episode')
        rewards = np.zeros(n_episodes)
        infos = dict()
        
        for episode in progress_bar:
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0

            # Reset the eligibility trace if applicable
            self.reset_eligibility_trace()
            episode_infos = []

            while not done:
                next_state, reward, done, info = self.env.step(action)
                episode_infos.append(info)
                next_action = self.choose_action(next_state)
                
                self.learn(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            rewards[episode] = total_reward
            infos[episode] = episode_infos
            self.update_epsilon()
            progress_bar.set_postfix({'Total reward': total_reward, 'Epsilon': self.epsilon})
        return np.array(rewards), infos

    # Test the agent
    def test(self, n_episodes):
        all_rewards = []
        infos = []
        for episode in tqdm(range(n_episodes), desc='Testing', unit='episode'):
            state = self.env.reset()
            done = False
            rewards = []
            episode_infos = []
            while not done:
                action = self.choose_action(state)
                state, reward, done, info = self.env.step(action)
                rewards.append(reward)
                episode_infos.append(info)
            rewards = np.array(rewards)
            all_rewards.append(rewards)
            infos.append(episode_infos)
        return np.array(all_rewards).mean(axis=0), infos




class LearningUpdates:

    # Q-Learning Update Rule
    @staticmethod
    def q_learning_update(state, action, reward, next_state, next_action, done, value_function, alpha, gamma):
        q_values = value_function.get_q_values(state)
        next_q_values = value_function.get_q_values(next_state)
        
        best_next_action = np.argmax(next_q_values)
        td_target = reward + (gamma * next_q_values[best_next_action] if not done else reward)
        td_error = td_target - q_values[action]
        
        value_function.update(state, action,  alpha * td_error)


    # SARSA Update Rule
    @staticmethod
    def sarsa_update(state, action, reward, next_state, next_action, done, value_function, alpha, gamma):
        q_values = value_function.get_q_values(state)
        next_q_values = value_function.get_q_values(next_state)
        
        td_target = reward + (gamma * next_q_values[next_action] if not done else reward)
        td_error = td_target - q_values[action]
        
        value_function.update(state, action, alpha * td_error)

    # Expected SARSA Update Rule
    @staticmethod
    def expected_sarsa_update(state, action, reward, next_state, next_action, done, value_function, alpha, gamma, epsilon = 0.8):
        q_values = value_function.get_q_values(state)
        next_q_values = value_function.get_q_values(next_state)
        
        
        action_probabilities = np.ones_like(next_q_values) * (epsilon / len(next_q_values))
        best_action = np.argmax(next_q_values)
        action_probabilities[best_action] += 1 - epsilon

        expected_next_q = np.dot(next_q_values, action_probabilities)
        
        td_target = reward + (gamma * expected_next_q if not done else reward)
        td_error =  td_target - q_values[action]
        
        value_function.update(state, action, alpha * td_error)


    # TD(n) Update Rule (for simplicity, we use n=1 for TD(1), which is similar to SARSA)
    @staticmethod
    def td_n_update(state, action, reward, next_state, next_action, done, value_function, alpha, gamma, n=1, trajectory=None):
        """
        TD(n) update for n-step returns.
        
        :param state: Current state
        :param action: Current action
        :param reward: Immediate reward received
        :param next_state: Next state after taking the action
        :param next_action: Next action (not used in TD(n), but kept for compatibility)
        :param done: Whether the episode has ended
        :param value_function: The value function (approximator) to update Q-values
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param n: Number of steps to consider in TD(n)
        :param trajectory: A deque storing the trajectory of (state, action, reward)
        """
        if trajectory is None:
            trajectory = deque(maxlen=n)

        # Store the current (state, action, reward) in the trajectory
        trajectory.append((state, action, reward))
        
        # If we have enough elements in the trajectory, calculate the n-step return
        if len(trajectory) == n or done:
            # Calculate the n-step return (or truncated return if done)
            n_step_return = sum([gamma**i * traj[2] for i, traj in enumerate(trajectory)])  # Sum of discounted rewards
            
            if not done:
                next_q_values = value_function.get_q_values(next_state)
                n_step_return += gamma**n * np.max(next_q_values)  # Add the estimated value of the next state

            # Get the first (state, action) in the trajectory (i.e., the state-action pair to update)
            first_state, first_action, _ = trajectory[0]

            # Get Q-values for the first state
            q_values = value_function.get_q_values(first_state)
            td_error = n_step_return - q_values[first_action]
            
            # Update the value function for the first state-action pair
            value_function.update(first_state, first_action, alpha * td_error)

            # Remove the first element from the trajectory to move forward
            trajectory.popleft()

        return trajectory  # Return the updated trajectory
    
    #Q-Learning(λ) Update Rule
    @staticmethod
    def q_lambda_update(states, action, reward, next_states, next_action, done, value_function, alpha, gamma, el_decay, eligibility_trace, forget_threshold=1e-6):
        actions = list(range(10))
        q_values = value_function.get_q_values(states, actions)
        next_q_values = value_function.get_q_values(next_states, actions)

        best_next_action = np.argmax(next_q_values)
        td_error = reward + (gamma * next_q_values[best_next_action] if not done else reward) - q_values[action]
        
        state_tuple = ( tuple(state) for state in states)
        eligibility_trace[(state_tuple, action)] = eligibility_trace.get((state_tuple, action), 0) + 1
        gamma_el_decay = gamma * el_decay
        keys_to_delete = []

        # Iterate over all the state-action pairs in the eligibility trace
        for (s, a), trace_value in eligibility_trace.items():
            state_couple = [ np.array(list(state)) for state in s]
            state_couple = [ np.append(state, a) for state in state_couple]

            # Update the value function for each (state, action) pair
            value_function.update(
                states=state_couple,
                target=alpha * td_error * trace_value,
                alpha=alpha
            )
            eligibility_trace[(s, a)] *= gamma_el_decay if a == best_next_action else 0

            if eligibility_trace[(s, a)] < forget_threshold:
                keys_to_delete.append((s, a))

        # Delete the state-action pairs with trace values below the threshold
        for key in keys_to_delete:
            del eligibility_trace[key]

        return eligibility_trace
    
    #Sarsa(λ) Update Rule
    @staticmethod
    def sarsa_lambda_update(state, action, reward, next_state, next_action, done, value_function, alpha, gamma, el_decay, eligibility_trace, forget_threshold=1e-6):
        actions = list(range(10))
        q_values = value_function.get_q_values(state, actions)
        next_q_values = value_function.get_q_values(next_state, actions)

        td_target = reward + (gamma * next_q_values[next_action] if not done else reward)
        td_error = td_target - q_values[action]  # Temporal difference error (delta)

        state_tuple = ( tuple(s) for s in state)
        eligibility_trace[(state_tuple, action)] = eligibility_trace.get((state_tuple, action), 0) + 1
        gamma_el_decay = gamma * el_decay

        # Iterate over all the state-action pairs in the eligibility trace
        key_to_delete = []
        for (s, a), trace_value in eligibility_trace.items():
            state_couple = [ np.array(list(_s)) for _s in s]
            state_couple = [ np.append(_s, a) for _s in state_couple]
            # print(state_couple.shape)
            value_function.update(
                states=state_couple,
                target=alpha * td_error * trace_value,
                alpha=alpha
            )
            
            # Decay eligibility trace for each (state, action) pair
            eligibility_trace[(s, a)] *= gamma_el_decay

            # Remove the state-action pair from the eligibility trace if the trace value is below a threshold
            if eligibility_trace[(s, a)] < forget_threshold:
                key_to_delete.append((s, a))

        # Delete the state-action pairs with trace values below the threshold
        for key in key_to_delete:
            del eligibility_trace[key]

        return eligibility_trace




class QLearningAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(env, value_function, LearningUpdates.q_learning_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        
class TDNAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, n=1):
        super().__init__(env, value_function, LearningUpdates.td_n_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.n = n
        
class SarsaAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(env, value_function, LearningUpdates.sarsa_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min)


### ISSUE: non riusciamo a passare ad ESA il parametro epsilon definito in LearningAgent, come si fa? Per adesso glielo passiamo direttamente in expected sarsa update
class ExpectedSarsaAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(env, value_function, LearningUpdates.expected_sarsa_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        # self.epsilon = epsilon
    

class SarsaLambdaAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, el_decay=0.8):
        super().__init__(env, value_function, LearningUpdates.sarsa_lambda_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min, el_decay)

class QLambdaAgent(LearningAgent):
    def __init__(self, env, value_function, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, el_decay=0.8):
        super().__init__(env, value_function, LearningUpdates.q_lambda_update, policies.epsilon_greedy_policy, alpha, gamma, epsilon, epsilon_decay, epsilon_min, el_decay)
        
