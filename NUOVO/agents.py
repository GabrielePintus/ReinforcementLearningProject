from generic_agent import GenericAgent
class QLearningAgent(GenericAgent):
    def __init__(
        self,
        action_size, # 10 in our case
        q_value_approximator, 
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        gamma=0.99,
        logger=None
    ):
        super().__init__(action_size, q_value_approximator, epsilon, epsilon_decay, min_epsilon, gamma, logger)

    def target_policy(self, state):
        return self.greedy_policy(state)
    
    def behaviour_policy(self, state):
        return self.epsilon_greedy_policy(state)
        

    
class SarsaAgent(GenericAgent):
    def __init__(
        self,
        action_size, # 10 in our case
        q_value_approximator, 
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        gamma=0.99,
        logger=None
    ):
        super().__init__(action_size, q_value_approximator, epsilon, epsilon_decay, min_epsilon, gamma, logger)

    def target_policy(self, state):
        return self.epsilon_greedy_policy(state)

    def behaviour_policy(self, state):
        return self.epsilon_greedy_policy(state)
    


    



        
        
 
    
 