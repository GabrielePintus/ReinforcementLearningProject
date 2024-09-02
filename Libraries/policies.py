import numpy as np




class Policy:

    def __init__(
            self,
            value_function,
            env,
        ):
        self.value_function = value_function
        self.env = env

    def get_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __call__(self, state):
        return self.get_action(state)



class EpsilonGreedyPolicy(Policy):
    
        def __init__(
                self,
                value_function,
                env,
                epsilon=0.5,
                epsilon_decay=0.9,
                epsilon_min=0.1
            ):
            super().__init__(value_function, env)
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
    
        def get_action(self, state):
            actions = list(range(self.env.action_space.n))
            if np.random.random() < self.epsilon:
                return np.random.choice(actions)  # Explore
            else:
                q_values = self.value_function.get_q_values(state, actions)
                max_value = np.max(list(q_values.values()))

                max_keys = [key for key, value in q_values.items() if value == max_value]
                random_max_key = np.random.choice(max_keys)

                return random_max_key
    
        def update(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min



if __name__ == '__main__':
    # Example dictionary
    arr = {0: 3, 1: 1, 2: 3, 3: 2}

    # Step 1: Find the maximum value
    max_value = max(arr.values())

    # Step 2: Get all keys where the maximum value occurs
    max_keys = [key for key, value in arr.items() if value == max_value]

    # Step 3: Randomly select one of these keys
    random_max_key = np.random.choice(max_keys)

    print("Selected key:", random_max_key)