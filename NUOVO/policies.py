import numpy as np



class EpsilonGreedyPolicy:
    
        def __init__(
                self,
                action_space,
                value_function,
                env,
                epsilon=0.5,
                epsilon_decay=0.9,
                epsilon_min=0.0
            ):
            super().__init__(value_function, env)
            self.action_space = action_space

            # Value function
            self.value_function = value_function
            
            # Exploration
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            
    
        def get_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space)
            else:
                values = self.value_function(state)
                return np.argmax(values)

    
        def update(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

        def __call__(self, state):
            return self.get_action(state)



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