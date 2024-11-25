import numpy as np



class LearningAgent:

    def __init__(
        self,
        env,
        value_function,
        update_rule,
        policy,
        alpha=0.1,
        gamma=0.99,
        el_decay=None
    ):
        """
        Generalized reinforcement learning agent.

        Args:
            env: The environment instance (must have reset() and step() methods).
            value_function: Value function approximator (e.g., table or neural network).
            update_rule: Function for updating the value function (e.g., Q-learning).
            policy: Policy object for action selection (e.g., epsilon-greedy).
            alpha: Learning rate.
            gamma: Discount factor.
            el_decay: Lambda for eligibility traces (default: None, no traces).
        """
        self.env = env
        self.update_rule = update_rule  # Generalized update rule (e.g., Q-learning, SARSA)
        self.policy = policy  # Generalized policy (e.g., epsilon-greedy)
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.el_decay = el_decay    # eligibility traces lambda decay term
        self.eligibility_trace = None

        # Initialize eligibility trace for SARSA(λ) if lambda is provided, 
        if el_decay is not None:
            self.reset_eligibility_trace()

    def reset_eligibility_trace(self):
        if self.el_decay is not None:
            self.eligibility_trace = {}
    
    # Update the value function using the specified update rule
    def learn(self, state, action, reward, next_state, next_action, done):
        if self.el_decay is not None:
            self.update_rule(state, action, reward, next_state, next_action, done, self.value_function, self.alpha, self.gamma, self.el_decay, self.eligibility_trace)
        else:
            self.update_rule(state, action, reward, next_state, next_action, done, self.value_function, self.alpha, self.gamma)


    def train(self, n_episodes):
        progress_bar = range(n_episodes)
        rewards = np.zeros(n_episodes)

        for episode in progress_bar:
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0

            # Reset the eligibility trace if applicable
            self.reset_eligibility_trace()

            while not done:
                next_state, reward, done = self.env.step(action)

                # Choose next action
                next_action = self.policy(next_state)

                # Learn from the transition
                self.learn(state, action, reward, next_state, next_action, done)

                # Update the state and action
                state = next_state
                action = next_action
                total_reward += reward

            rewards[episode] = total_reward
            self.policy.update()
        
        return rewards
    






class LearningUpdates:

    @staticmethod
    def q_learning(
        state, action, reward, next_state, _, done, value_function, alpha, gamma, **kwargs
    ):
        """
        Q-Learning update rule.

        Args:
            state: Current state.
            action: Current action.
            reward: Reward received.
            next_state: Next state.
            _: Placeholder for next_action (not used in Q-Learning).
            done: Whether the episode is done.
            value_function: Value function table or approximator.
            alpha: Learning rate.
            gamma: Discount factor.
        """
        # Get the current Q-value
        q_value = value_function(state)[action]

        # Compute the TD target
        td_target = reward + gamma * np.max(value_function(next_state)) * (1 - done)

        # Compute the TD error
        td_error = td_target - q_value

        # Update the Q-value
        value_function.update(state, action, alpha * td_error)







# Basic agent using SARSA update rule, can be changed to Q-learning by changing the update rule to LearningUpdates.q_learning_update

class QLearningAgent(LearningAgent):
    def __init__(self, env, value_function, policy, alpha=0.1, gamma=0.99):
        super().__init__(env, value_function, LearningUpdates.q_learning, policy, alpha, gamma)

    def reset_eligibility_trace(self):
        pass