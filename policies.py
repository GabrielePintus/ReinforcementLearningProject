import numpy as np

class Policies:
    
    # Example Policy: Epsilon-Greedy
    @staticmethod
    def epsilon_greedy_policy(state, value_function, epsilon):
        if np.random.random() < epsilon:
            return value_function.env.action_space.sample()  # Explore
        else:
            q_values = value_function.get_q_values(state)
            return np.argmax(q_values)  # Exploit