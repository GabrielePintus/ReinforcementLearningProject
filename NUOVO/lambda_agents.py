from generic_agent import GenericAgent
from collections import defaultdict

class QLambdaAgent(GenericAgent):
    def __init__(
        self,
        action_size, # 10 in our case
        q_value_approximator, 
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        gamma=0.99,
        lam=0.85, # [0,1] 0: TD learning, 1: Monte Carlo
        trace_decay=0.9, #TODO: change values to be consinstent with the paper
        cutoff= 0.01, # parameter to discard negligible traces, change value according to specific needs
        logger=None
    ):
        super().__init__(action_size, q_value_approximator, epsilon, epsilon_decay, min_epsilon, gamma, logger)
        self.lam = lam
        self.trace_decay = trace_decay
        self.cutoff = cutoff
        self.eligibility_traces = None

    def reset_eligibility_trace(self):
        self.eligibility_traces = defaultdict(float)

    def target_policy(self, state):
        return self.greedy_policy(state)
    
    def behaviour_policy(self, state):
        return self.epsilon_greedy_policy(state)
    
    def train(self,env ,n_episodes=1, buffer_size=16):
        for _ in range(n_episodes):
            # Reset the environment
            state = env.reset()
            # We re-initialize the eligibility traces to zero
            self.reset_eligibility_trace() 
            done = False

            # Initialize the buffer for storing transitions
            buffer = []
            
            while not done:
                # Choose an action based on the behaviour policy (epsilon-greedy)
                action = self.behaviour_policy(state)
                
                # Take action and observe next state and reward
                next_state, reward, done, bankroll_change = env.update(action)

                # Choose the best action according to the target policy
                best_action = self.target_policy(next_state)
                best_state_action = self._combine_state_action(next_state, best_action)
                # Predict the Q-value
                state_representation = self._combine_state_action(state, action)  # TODO: is this already passed as an approximated repr with tilecoding? NO

                target = reward + self.gamma * self.q_value_approximator.predict(best_state_action)
                if state_representation not in self.eligibility_traces:
                    self.eligibility_traces[state_representation] = 1
                else:
                    self.eligibility_traces[state_representation] += 1

               

               # TODO: problem, we should update and decay traces for each observation, 
               # we aree using a buffer though, so we need to update the traces for the whole buffer
               # If my understanding is correct if we set the buffer to size 1, it's just like
                # updating the traces for each observation, and also the q-value approximator
            
            # What if we kept track of the traces directly in the approximator?
            # we just need the additional info about the action taken
            # we could also keep track of the traces in the buffer
            # and then somehow update at the end of the episode