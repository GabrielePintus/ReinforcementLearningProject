import numpy as np



class ReplayBuffer():
    '''
    The ReplayBuffer object stores the agent's experiences in a buffer. The agent samples from this buffer to learn during training.
    '''

    def __init__(self, max_size, input_shape, n_actions):
        '''
        Initializes the ReplayBuffer object.
        max_size: int, the maximum number of experiences the buffer can store.
        input_shape: tuple, the shape of the input data.
        n_actions: int, the number of possible actions.
        '''
        
        # Memory size and counter
        self.memory_size = max_size
        self.memory_counter = 0

        # Initialize the memory arrays for the dynamics function
        self.state_memory = np.zeros((self.memory_size, *input_shape))     # s_t
        self.new_state_memory = np.zeros((self.memory_size, *input_shape)) # s_t+1
        self.action_memory = np.zeros((self.memory_size, n_actions))       # a_t
        self.reward_memory = np.zeros(self.memory_size)                    # r_t

        # Boolean array to store whether the episode has ended
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    
    def __str__(self):
        '''
        Returns the string representation of the ReplayBuffer object.
        '''
        return f'ReplayBuffer(max_size={self.memory_size}, memory_counter={self.memory_counter})\n\n state_memory: {self.state_memory}\n\n new_state_memory: {self.new_state_memory}\n\n action_memory: {self.action_memory}\n\n reward_memory: {self.reward_memory}\n\n terminal_memory: {self.terminal_memory}'
    

    def store_transition(self, state, action, reward, next_state, done):
        '''
        Store the agent's experience in the buffer.
        state: the current state.
        action: the action taken.
        reward: the reward received.
        state_: the next state.
        done: whether the episode has ended.
        '''
        
        # Get the index of the memory to store the experience, it wraps around when it 
        # reaches the memory size, so basically a FIFO buffer
        index = self.memory_counter % self.memory_size

        # Store the experience
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size : int):
        '''
        Generate a random sample of experiences from the buffer.
        batch_size: the number of experiences to sample.
        return: the sampled experiences.
        '''

        # Ensure that we sample from the memory that has been filled
        max_mem = min(self.memory_counter, self.memory_size)

        # Randomly sample experiences from the memory
        batch = np.random.choice(max_mem, batch_size)

        # Save & return the sampled experiences
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones