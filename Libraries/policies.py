import numpy as np

def epsilon_greedy_policy(state, value_function, env, epsilon):
    """
    Epsilon-Greedy policy: With probability epsilon, choose a random action;
    otherwise, choose the action with the highest Q-value.
    
    :param state: Current state
    :param value_function: The value function (approximator) to get Q-values
    :param env: The environment object to sample random actions
    :param epsilon: Exploration probability
    :return: Action to take
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        actions = list(range(env.action_space.n))
        q_values = value_function.get_q_values(state, actions)
        return np.argmax(q_values)  # Exploit


def softmax_policy(state, value_function, env, temperature=1.0):
    """
    Softmax policy: Select actions based on their probability,
    which is proportional to the exponentiated Q-value.
    
    :param state: Current state
    :param value_function: The value function (approximator) to get Q-values
    :param env: The environment object to sample random actions
    :param temperature: Controls the randomness; higher values = more exploration
    :return: Action to take
    """
    q_values = value_function.get_q_values(state)
    preferences = q_values / temperature
    max_pref = np.max(preferences)
    exp_preferences = np.exp(preferences - max_pref)  # Subtract max to prevent overflow
    probabilities = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(len(q_values), p=probabilities)


def ucb_policy(state, value_function, env, action_counts, total_steps, c=2):
    """
    Upper Confidence Bound (UCB) policy: Selects actions based on the
    Q-value and the uncertainty (exploration term).
    
    :param state: Current state
    :param value_function: The value function (approximator) to get Q-values
    :param env: The environment object to sample random actions
    :param action_counts: A list of counts of how many times each action has been selected
    :param total_steps: The total number of steps taken
    :param c: Exploration coefficient (controls trade-off between exploration and exploitation)
    :return: Action to take
    """
    q_values = value_function.get_q_values(state)
    ucb_values = q_values + c * np.sqrt(np.log(total_steps + 1) / (np.array(action_counts) + 1e-5))
    return np.argmax(ucb_values)
