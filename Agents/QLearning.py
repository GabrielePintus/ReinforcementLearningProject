
from Agents.Agent import Agent


class QLearningAgent(Agent):
    def __init__(self, num_actions, epsilon=0.1, learning_rate=0.1, discount_factor=0.99):
        (super(QLearningAgent, self).__init__(epsilon))
        pass